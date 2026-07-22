"""Revisioned semantic field projection, independent from graph evidence."""

from __future__ import annotations

import hashlib
import math
from contextlib import nullcontext
from typing import Any, Mapping, Sequence

from .graph_repository import decode, encode, stable_id, utcnow
from .hybrid.contracts import QueryFrame, RetrievalHit, clamp
from .semantic_field_dynamics import SemanticFieldDynamics


FIELD_UNIVERSE = "global-semantic"
MAX_DISPLACEMENT = 0.05
MIN_DISTANCE = 0.08


def _seed(value: str) -> list[float]:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return [round((digest[index] / 255.0) * 2.0 - 1.0, 6) for index in range(3)]


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _covariance(radius: float = 0.25, dimensions: int = 3) -> list[list[float]]:
    return [[radius if row == column else 0.0 for column in range(dimensions)] for row in range(dimensions)]


class SemanticFieldService:
    def __init__(self, repository: Any) -> None:
        self.repository = repository
        self.dynamics = SemanticFieldDynamics(max_displacement=MAX_DISPLACEMENT, min_distance=MIN_DISTANCE)

    @staticmethod
    def _revision(conn: Any) -> int:
        return int(conn.execute("SELECT COALESCE(MAX(revision), 0) + 1 FROM field_revisions").fetchone()[0])

    @staticmethod
    def _source_events(conn: Any) -> list[Any]:
        return conn.execute(
            """SELECT p.id AS participant_id,m.entity_id,e.id AS event_id,e.source_id,
                      e.predicate_concept_id,e.predicate_lemma,e.polarity
               FROM graph_events e JOIN graph_participants p ON p.event_id=e.id
               JOIN graph_mentions m ON m.id=p.mention_id
               JOIN knowledge_sources s ON s.id=e.source_id
               WHERE s.status='CONFIRMED' ORDER BY e.source_id,e.id,p.ordinal_hint"""
        ).fetchall()

    def _write_revision(self, conn: Any, *, changed_source_id: str = "", reason: str = "INGEST") -> dict[str, Any]:
        rows = self._source_events(conn)
        revision = self._revision(conn)
        now = utcnow()
        event_groups: dict[str, list[dict[str, Any]]] = {}
        sources: dict[str, set[str]] = {}
        for row in rows:
            item = dict(row)
            event_groups.setdefault(str(item["event_id"]), []).append(item)
            sources.setdefault(str(item["source_id"]), set()).add(str(item["entity_id"]))
        concepts = sorted({str(row["entity_id"]) for row in rows})
        conn.execute(
            """INSERT INTO field_revisions(revision,based_on_event_revision,status,metrics_json,created_at)
               VALUES(?,?, 'PROPOSED',?,?,?)""".replace("VALUES(?,?, 'PROPOSED',?,?,?)", "VALUES(?,?, 'PROPOSED',?,?)"),
            (revision, int(conn.execute("SELECT value FROM graph_meta WHERE key='projection_revision'").fetchone()[0] or 0), encode({"reason": reason}), now),
        )
        active_dims = [str(item["id"]) for item in conn.execute(
            "SELECT id FROM latent_dimensions WHERE status IN ('active','shared','probation') ORDER BY stability DESC,id LIMIT 32"
        ).fetchall()]
        if concepts:
            conn.execute("DELETE FROM semantic_cloud_current_projections WHERE universe_id=? AND cloud_id NOT IN ({})".format(",".join("?" for _ in concepts)), (FIELD_UNIVERSE, *[stable_id("semantic-cloud", item) for item in concepts]))
        else:
            conn.execute("DELETE FROM semantic_cloud_current_projections WHERE universe_id=?", (FIELD_UNIVERSE,))
        previous: dict[str, list[float]] = {}
        for row in conn.execute("SELECT cloud_id,field_revision FROM semantic_cloud_current_projections WHERE universe_id=?", (FIELD_UNIVERSE,)).fetchall():
            old = conn.execute("SELECT applied_center_json FROM semantic_cloud_projection_revisions WHERE cloud_id=? AND universe_id=? AND field_revision=?", (row["cloud_id"], FIELD_UNIVERSE, row["field_revision"])).fetchone()
            previous[str(row["cloud_id"])] = decode(old[0], []) if old and old[0] else []
        if not concepts:
            conn.execute("UPDATE field_revisions SET status='VALIDATED',metrics_json=? WHERE revision=?", (encode({"updated_clouds": 0}), revision))
            conn.execute("UPDATE field_revisions SET status='APPLIED',applied_at=? WHERE revision=?", (now, revision))
            return {"field_revision": revision, "updated_cloud_ids": [], "status": "APPLIED", "removed_source_id": changed_source_id}

        centers: dict[str, list[float]] = {}
        cloud_ids = {concept: stable_id("semantic-cloud", concept) for concept in concepts}
        for concept in concepts:
            cloud_id = cloud_ids[concept]
            cloud_row = conn.execute("SELECT * FROM semantic_clouds WHERE id=?", (cloud_id,)).fetchone()
            bootstrap = decode(cloud_row["bootstrap_center_json"], []) if cloud_row and "bootstrap_center_json" in cloud_row.keys() else _seed(concept)
            if len(bootstrap) != 3:
                bootstrap = _seed(concept)
            support = [other for other in concepts if other != concept and any(concept in values and other in values for values in sources.values())]
            if support:
                # Learned geometry is derived from coactivation structure. The hash
                # seed is retained only as the initial placement for an isolated cloud.
                vectors = [centers.get(item, _seed(item)) for item in support]
                learned = [sum(vector[index] for vector in vectors) / len(vectors) for index in range(3)]
                status = "LEARNED"
            else:
                learned = previous.get(cloud_id) or bootstrap
                status = "BOOTSTRAP" if not previous.get(cloud_id) else "STABLE"
            centers[concept] = [round(float(value), 6) for value in learned]
            observed_count = sum(concept in values for values in sources.values())
            mass = min(1.0, 0.2 + 0.04 * observed_count)
            density = min(1.0, 0.2 + 0.03 * observed_count)
            stability = min(1.0, 0.2 + 0.02 * observed_count)
            conn.execute(
                """INSERT INTO semantic_clouds(id,cloud_type,concept_id,mass,density,halo,stability,permeability,
                   bootstrap_center_json,learned_center_json,active_dimensions_json,position_status,provenance_json,created_at,updated_at)
                   VALUES(?,?,?, ?,?,?,?, ?,?,?,?,?,?,?,?)
                   ON CONFLICT(concept_id) DO UPDATE SET mass=excluded.mass,density=excluded.density,
                   stability=excluded.stability,bootstrap_center_json=excluded.bootstrap_center_json,
                   learned_center_json=excluded.learned_center_json,active_dimensions_json=excluded.active_dimensions_json,
                   position_status=excluded.position_status,updated_at=excluded.updated_at""",
                (cloud_id, "concept", concept, mass, density, 0.35, stability, 0.5, encode(bootstrap), encode(centers[concept]), encode(active_dims), status, encode([{"source_id": source} for source, values in sources.items() if concept in values]), now, now),
            )
            for event_id, members in event_groups.items():
                for member in members:
                    if str(member["entity_id"]) != concept:
                        continue
                    context_key = f"predicate:{member['predicate_concept_id']}"
                    context_id = stable_id("contextual-cloud", cloud_id, context_key)
                    context_center = [round(value + seed * 0.015, 6) for value, seed in zip(centers[concept], _seed(context_key))]
                    conn.execute(
                        """INSERT INTO contextual_cloud_projections(id,cloud_id,context_key,center_json,covariance_json,activation_count,field_revision)
                           VALUES(?,?,?,?,?,1,?)
                           ON CONFLICT(cloud_id,context_key) DO UPDATE SET activation_count=activation_count+1,center_json=excluded.center_json,covariance_json=excluded.covariance_json,field_revision=excluded.field_revision""",
                        (context_id, cloud_id, context_key, encode(context_center), encode(_covariance(0.18)), revision),
                    )
            for dimension_index, dimension_id in enumerate(active_dims):
                coordinate = centers[concept][dimension_index % 3]
                conn.execute(
                    """INSERT INTO cloud_dimension_projections(cloud_id,dimension_id,coordinate,variance,confidence,source_support_json,revision)
                       VALUES(?,?,?,?,?,?,?)""",
                    (cloud_id, dimension_id, coordinate, 0.1, min(1.0, 0.4 + 0.1 * len(sources)), encode(sorted(sources)), revision),
                )

        for cloud_id, center in ((cloud_ids[item], centers[item]) for item in concepts):
            # Rebuilds are deterministic functions of the active Event Graph;
            # historical current coordinates are never used as semantic input.
            prior = list(center)
            cloud = conn.execute("SELECT * FROM semantic_clouds WHERE id=?", (cloud_id,)).fetchone()
            neighbours = []
            for other in concepts:
                if other == next(item for item in concepts if cloud_ids[item] == cloud_id):
                    continue
                neighbours.append({"cloud_id": cloud_ids[other], "center": centers[other], "learned_center": centers[other], "mass": 0.2, "coactivation_weight": 0.2})
            dynamics = self.dynamics.calculate({"cloud_id": cloud_id, "center": center, "learned_center": center, "mass": cloud["mass"], "stability": cloud["stability"], "density": cloud["density"], "halo": cloud["halo"]}, neighbours)
            proposed = [round(prior[index] + dynamics.limited_displacement[index], 6) for index in range(3)]
            principal_axes = [{"dimension_id": dimension_id, "axis": [1.0 if axis == index % 3 else 0.0 for axis in range(3)], "variance": 0.1} for index, dimension_id in enumerate(active_dims)] or [{"dimension_id": "display:x", "axis": [1.0, 0.0, 0.0], "variance": 0.25}, {"dimension_id": "display:y", "axis": [0.0, 1.0, 0.0], "variance": 0.25}, {"dimension_id": "display:z", "axis": [0.0, 0.0, 1.0], "variance": 0.25}]
            transition = {"previous_center": prior, "proposed_center": proposed, "applied_center": proposed, "force_components": [item.as_dict() for item in dynamics.forces], "supporting_event_ids": [event_id for event_id, members in event_groups.items() if any(str(row["entity_id"]) == next(item for item in concepts if cloud_ids[item] == cloud_id) for row in members)], "supporting_source_ids": sorted(sources), "validation": {"passed": True, "minimum_distance": MIN_DISTANCE}, "max_displacement_applied": _distance(prior, proposed) >= MAX_DISPLACEMENT * 0.99}
            conn.execute("""INSERT INTO semantic_cloud_projection_revisions(cloud_id,universe_id,field_revision,previous_center_json,proposed_center_json,applied_center_json,covariance_json,principal_axes_json,active_dimensions_json,validation_json,status,created_at,applied_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,'PROPOSED',?,?)""", (cloud_id, FIELD_UNIVERSE, revision, encode(prior), encode(proposed), encode(proposed), encode(_covariance()), encode(principal_axes), encode(active_dims), encode({}), now, None))
            conn.execute("UPDATE semantic_cloud_projection_revisions SET status='VALIDATED',validation_json=? WHERE cloud_id=? AND universe_id=? AND field_revision=?", (encode(transition["validation"]), cloud_id, FIELD_UNIVERSE, revision))
            conn.execute("UPDATE semantic_cloud_projection_revisions SET status='APPLIED',applied_at=?,applied_center_json=? WHERE cloud_id=? AND universe_id=? AND field_revision=?", (now, encode(proposed), cloud_id, FIELD_UNIVERSE, revision))
            conn.execute("INSERT OR REPLACE INTO semantic_cloud_projections(cloud_id,universe_id,center_json,covariance_json,principal_axes_json,local_potential,field_revision) VALUES(?,?,?,?,?,?,?)", (cloud_id, FIELD_UNIVERSE, encode(proposed), encode(_covariance()), encode(principal_axes), -0.2, revision))
            conn.execute("INSERT OR REPLACE INTO semantic_cloud_current_projections(cloud_id,universe_id,field_revision) VALUES(?,?,?)", (cloud_id, FIELD_UNIVERSE, revision))
            conn.execute("INSERT INTO field_transitions(id,revision,source_id,cloud_id,transition_type,payload_json,created_at) VALUES(?,?,?,?,?,?,?)", (stable_id("field-transition", revision, cloud_id), revision, changed_source_id or "rebuild", cloud_id, "DISPLACEMENT", encode(transition), now))
            for force in dynamics.forces:
                conn.execute("INSERT INTO semantic_field_force_traces(id,field_revision,cloud_id,force_type,source_cloud_id,vector_json,magnitude,evidence_ids_json,payload_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)", (stable_id("field-force", revision, cloud_id, force.type, force.source_cloud_id or "self"), revision, cloud_id, force.type, force.source_cloud_id, encode(list(force.vector)), force.magnitude, encode(list(force.evidence_ids)), encode(force.payload or {}), now))
        conn.execute("DELETE FROM field_source_contributions WHERE source_revision=?", (revision,))
        for source_id, source_concepts in sources.items():
            for event_id, members in event_groups.items():
                event_concepts = {str(item["entity_id"]) for item in members}
                for concept in sorted(event_concepts):
                    cloud_id = cloud_ids[concept]
                    context_ids = [stable_id("contextual-cloud", cloud_id, f"predicate:{item['predicate_concept_id']}") for item in members if str(item["entity_id"]) == concept]
                    conn.execute("INSERT INTO field_source_contributions(source_id,source_revision,event_id,cloud_id,mass_delta,density_delta,force_delta_json,shape_delta_json,context_projection_ids_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)", (source_id, revision, event_id, cloud_id, 0.04, 0.03, encode([]), encode({}), encode(context_ids), now))
        conn.execute("UPDATE field_revisions SET status='VALIDATED',metrics_json=? WHERE revision=?", (encode({"updated_clouds": len(concepts), "source_count": len(sources)}), revision))
        conn.execute("UPDATE field_revisions SET status='APPLIED',applied_at=? WHERE revision=?", (now, revision))
        return {"field_revision": revision, "updated_cloud_ids": [cloud_ids[item] for item in concepts], "status": "APPLIED", "source_count": len(sources)}

    def ingest_source(self, source_id: str, *, connection: Any = None) -> dict[str, Any]:
        transaction = self.repository.transaction() if connection is None else nullcontext(connection)
        with transaction as conn:
            source = conn.execute("SELECT status FROM knowledge_sources WHERE id=?", (source_id,)).fetchone()
            if not source or str(source["status"]) != "CONFIRMED":
                return {"source_id": source_id, "status": "SKIPPED", "reason": "SOURCE_NOT_CONFIRMED"}
            result = self._write_revision(conn, changed_source_id=source_id, reason="INGEST")
            result["source_id"] = source_id
            return result

    def remove_source_contribution(self, source_id: str, *, connection: Any = None) -> dict[str, Any]:
        transaction = self.repository.transaction() if connection is None else nullcontext(connection)
        with transaction as conn:
            result = self._write_revision(conn, changed_source_id=source_id, reason="REMOVE_SOURCE_CONTRIBUTION")
            result.update({"source_id": source_id, "removed": True})
            return result

    def rebuild_from_event_revision(self, event_revision: int | None = None, *, connection: Any = None) -> dict[str, Any]:
        transaction = self.repository.transaction() if connection is None else nullcontext(connection)
        with transaction as conn:
            return self._write_revision(conn, reason=f"EVENT_REVISION:{event_revision or 0}")

    def project_query(self, frame: QueryFrame) -> dict[str, Any]:
        elements = [item for item in frame.known_elements if isinstance(item, Mapping)]
        terms = []
        for item in frame.known_elements:
            if isinstance(item, Mapping):
                terms.extend(str(item.get(key) or "") for key in ("concept_id", "entity_id", "lemma", "surface", "value"))
            else:
                terms.append(str(item))
        terms = [item.casefold() for item in terms if item]
        with self.repository.transaction() as conn:
            rows = conn.execute("""SELECT c.*,p.center_json,p.covariance_json,p.field_revision,p.principal_axes_json
               FROM semantic_clouds c JOIN semantic_cloud_current_projections cp ON cp.cloud_id=c.id AND cp.universe_id=?
               JOIN semantic_cloud_projections p ON p.cloud_id=cp.cloud_id AND p.universe_id=cp.universe_id AND p.field_revision=cp.field_revision""", (FIELD_UNIVERSE,)).fetchall()
            entities = {str(row["id"]): str(row["canonical_lemma"]) for row in conn.execute("SELECT id,canonical_lemma FROM graph_entities").fetchall()}
        anchors = []
        for row in rows:
            concept = str(row["concept_id"])
            lemma = entities.get(concept, concept).casefold()
            match_index = next((index for index, term in enumerate(terms) if term == concept.casefold() or term == lemma or term in lemma), None)
            if match_index is None:
                continue
            source = "CONCEPT_ID" if terms[match_index] == concept.casefold() else ("LEMMA" if terms[match_index] == lemma else "CONTEXT")
            anchors.append({"cloud_id": str(row["id"]), "concept_id": concept, "center": decode(row["center_json"], []), "covariance": decode(row["covariance_json"], []), "field_revision": int(row["field_revision"]), "resolution_source": source, "confidence": 1.0 if source == "CONCEPT_ID" else 0.72 if source == "LEMMA" else 0.35, "query_element_id": str(elements[match_index].get("id") or elements[match_index].get("element_id") or terms[match_index]) if match_index < len(elements) and isinstance(elements[match_index], Mapping) else terms[match_index]})
        center = [sum(item["center"][index] for item in anchors) / len(anchors) for index in range(3)] if anchors else []
        negatives = list(frame.negations) + list(frame.exclusions)
        active_dimensions = sorted({dimension for item in anchors for dimension in decode(next((row["active_dimensions_json"] for row in rows if str(row["id"]) == item["cloud_id"]), "[]"), [])})
        return {"anchor_clouds": anchors, "positive_gradients": [{"cloud_id": item["cloud_id"], "weight": item["confidence"], "resolution_source": item["resolution_source"]} for item in anchors], "negative_gradients": [{"value": item, "type": "REPULSION_CONSTRAINT", "excluded_region": True} for item in negatives], "relation_projections": [{"relation": gap.expected_relation, "gap_id": gap.gap_id, "weight": 0.5} for gap in frame.gaps if gap.expected_relation], "gap_field": {"gap_ids": [gap.gap_id for gap in frame.gaps]}, "temporal_projection": frame.temporal_scope or {}, "context_projection": {"known_elements": list(frame.known_elements), "contextual": bool(frame.context_inheritance.get("mode") == "ALLOW")}, "active_dimensions": active_dimensions, "field_region": {"center": center, "radius": 0.75, "active_dimensions": active_dimensions, "field_revision": max((item["field_revision"] for item in anchors), default=0)}}

    def neighbourhood(self, projection: Mapping[str, Any], *, limit: int = 32) -> list[RetrievalHit]:
        region = projection.get("field_region") if isinstance(projection.get("field_region"), Mapping) else {}
        center = list(region.get("center") or [])
        if len(center) != 3:
            return []
        anchors = {str(item.get("cloud_id")) for item in projection.get("anchor_clouds") or () if isinstance(item, Mapping)}
        with self.repository.transaction() as conn:
            rows = conn.execute("SELECT c.*,p.center_json,p.covariance_json,p.field_revision FROM semantic_clouds c JOIN semantic_cloud_current_projections cp ON cp.cloud_id=c.id AND cp.universe_id=? JOIN semantic_cloud_projections p ON p.cloud_id=cp.cloud_id AND p.universe_id=cp.universe_id AND p.field_revision=cp.field_revision", (FIELD_UNIVERSE,)).fetchall()
        hits = []
        for row in rows:
            if str(row["id"]) in anchors:
                continue
            cloud_center = decode(row["center_json"], [])
            if len(cloud_center) != 3:
                continue
            radius = 0.75 + float(row["halo"])
            distance = _distance(center, cloud_center)
            if distance > radius:
                continue
            hits.append(RetrievalHit(hit_id=stable_id("field-hit", row["id"], row["field_revision"]), element_id=str(row["id"]), element_type="cloud", source_id="semantic_field", match_score=clamp((1.0 - distance / max(MIN_DISTANCE, radius)) * (0.5 + 0.5 * float(row["stability"]))), matched_features=("FIELD_NEIGHBOURHOOD_RETRIEVAL",), payload={"cloud_id": str(row["id"]), "concept_id": str(row["concept_id"]), "center": cloud_center, "covariance": decode(row["covariance_json"], []), "mass": float(row["mass"]), "density": float(row["density"]), "halo": float(row["halo"]), "stability": float(row["stability"]), "field_revision": int(row["field_revision"]), "spatial_support": True}, provenance=({"source_type": "FIELD", "field_revision": int(row["field_revision"])},), retrieval_path=("field_projection", str(row["id"])), origin="FIELD"))
        return sorted(hits, key=lambda item: (-item.match_score, item.element_id))[:max(1, limit)]

    def snapshot(self, *, limit: int = 200, field_revision: int | None = None) -> dict[str, Any]:
        with self.repository.transaction() as conn:
            if field_revision is None:
                current = conn.execute("SELECT MAX(field_revision) FROM semantic_cloud_current_projections WHERE universe_id=?", (FIELD_UNIVERSE,)).fetchone()
                field_revision = int(current[0]) if current and current[0] is not None else 0
            revision = conn.execute("SELECT revision,based_on_event_revision,status,metrics_json,created_at,applied_at FROM field_revisions WHERE revision=?", (field_revision,)).fetchone()
            revision_number = int(revision["revision"]) if revision else 0
            rows = conn.execute("""SELECT c.*,p.applied_center_json AS center_json,p.covariance_json,p.principal_axes_json,p.active_dimensions_json,p.field_revision
               FROM semantic_clouds c JOIN semantic_cloud_projection_revisions p ON p.cloud_id=c.id AND p.universe_id=? AND p.field_revision=? AND p.status='APPLIED'
               ORDER BY c.stability DESC,c.mass DESC,c.id LIMIT ?""", (FIELD_UNIVERSE, revision_number, max(1, min(int(limit), 2000)))).fetchall() if revision_number else []
        return {"universe_id": FIELD_UNIVERSE, "field_revision": revision_number, "revision": {"based_on_event_revision": int(revision["based_on_event_revision"]), "status": str(revision["status"]), "metrics": decode(revision["metrics_json"], {}), "created_at": str(revision["created_at"]), "applied_at": revision["applied_at"]} if revision else None, "projection_method": "learned_sparse_dimensions_with_display_projection", "clouds": [{"cloud_id": str(row["id"]), "cloud_type": str(row["cloud_type"]), "concept_id": str(row["concept_id"]), "bootstrap_center": decode(row["bootstrap_center_json"], []), "learned_center": decode(row["learned_center_json"], []), "center": decode(row["center_json"], []), "display_center": decode(row["center_json"], []), "position_status": str(row["position_status"]), "active_dimensions": decode(row["active_dimensions_json"], []), "shape": {"type": "ellipsoid", "covariance": decode(row["covariance_json"], []), "principal_axes": decode(row["principal_axes_json"], [])}, "mass": float(row["mass"]), "density": float(row["density"]), "halo": float(row["halo"]), "stability": float(row["stability"]), "permeability": float(row["permeability"]), "field_revision": int(row["field_revision"])} for row in rows]}

    def restore_revision(self, field_revision: int, *, connection: Any = None) -> dict[str, Any]:
        transaction = self.repository.transaction() if connection is None else nullcontext(connection)
        with transaction as conn:
            exists = conn.execute("SELECT 1 FROM field_revisions WHERE revision=? AND status='APPLIED'", (int(field_revision),)).fetchone()
            if not exists:
                raise KeyError(field_revision)
            rows = conn.execute("SELECT cloud_id FROM semantic_cloud_projection_revisions WHERE universe_id=? AND field_revision=? AND status='APPLIED'", (FIELD_UNIVERSE, int(field_revision))).fetchall()
            conn.execute("DELETE FROM semantic_cloud_current_projections WHERE universe_id=?", (FIELD_UNIVERSE,))
            conn.executemany("INSERT INTO semantic_cloud_current_projections(cloud_id,universe_id,field_revision) VALUES(?,?,?)", [(str(row["cloud_id"]), FIELD_UNIVERSE, int(field_revision)) for row in rows])
            for row in conn.execute("SELECT cloud_id,applied_center_json,covariance_json,principal_axes_json FROM semantic_cloud_projection_revisions WHERE universe_id=? AND field_revision=? AND status='APPLIED'", (FIELD_UNIVERSE, int(field_revision))).fetchall():
                conn.execute("INSERT OR REPLACE INTO semantic_cloud_projections(cloud_id,universe_id,center_json,covariance_json,principal_axes_json,local_potential,field_revision) VALUES(?,?,?,?,?,?,?)", (str(row["cloud_id"]), FIELD_UNIVERSE, row["applied_center_json"], row["covariance_json"], row["principal_axes_json"], -0.2, int(field_revision)))
            return {"status": "ROLLED_BACK", "field_revision": int(field_revision), "cloud_count": len(rows)}
