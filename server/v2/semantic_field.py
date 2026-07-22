"""Revisioned semantic field projection, independent from graph evidence."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
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
                      e.predicate_concept_id,e.predicate_lemma,e.polarity,
                      p.observation_signature_json,m.preposition,e.properties_json
               FROM graph_events e JOIN graph_participants p ON p.event_id=e.id
               JOIN graph_mentions m ON m.id=p.mention_id
               JOIN knowledge_sources s ON s.id=e.source_id
               WHERE s.status='CONFIRMED' ORDER BY e.source_id,e.id,p.ordinal_hint"""
        ).fetchall()

    @staticmethod
    def _source_revision(conn: Any, source_id: str) -> int:
        row = conn.execute(
            "SELECT metadata_json FROM knowledge_sources WHERE id=?",
            (source_id,),
        ).fetchone()
        metadata = decode(row[0], {}) if row else {}
        raw = metadata.get("source_revision") or metadata.get("revision") or 1
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _dimension_coordinates(conn: Any, concepts: Sequence[str]) -> dict[str, dict[str, float]]:
        """Resolve graph concepts into canonical Universe projections.

        Semantic dimensions are stored sparsely.  The three display coordinates
        remain a projection only and are never reused as latent coordinates.
        """
        if not concepts:
            return {}
        placeholders = ",".join("?" for _ in concepts)
        rows = conn.execute(
            f"""SELECT ge.id AS concept_id,p.dimension_id,
                       AVG(p.membership) AS coordinate,
                       AVG(p.confidence) AS confidence
                FROM graph_entities ge
                JOIN lexemes l ON l.canonical_lemma=ge.canonical_lemma
                JOIN projections p
                  ON p.source_type='entity' AND p.source_id=l.lexeme_entity_id
                JOIN latent_dimensions d ON d.id=p.dimension_id
                WHERE ge.id IN ({placeholders})
                  AND d.status IN ('probation','active','shared')
                GROUP BY ge.id,p.dimension_id
                ORDER BY ge.id,p.dimension_id""",
            list(concepts),
        ).fetchall()
        result: dict[str, dict[str, float]] = defaultdict(dict)
        for row in rows:
            result[str(row["concept_id"])][str(row["dimension_id"])] = round(
                float(row["coordinate"]), 6
            )
        return dict(result)

    def _write_revision(
        self,
        conn: Any,
        *,
        changed_source_id: str = "",
        reason: str = "INGEST",
    ) -> dict[str, Any]:
        raw_rows = [dict(row) for row in self._source_events(conn)]
        rows = [row for row in raw_rows if row.get("entity_id")]
        revision = self._revision(conn)
        now = utcnow()

        event_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        concept_sources: dict[str, set[str]] = defaultdict(set)
        concept_events: dict[str, set[str]] = defaultdict(set)
        source_events: dict[str, set[str]] = defaultdict(set)
        for item in rows:
            event_id = str(item["event_id"])
            source_id = str(item["source_id"])
            concept_id = str(item["entity_id"])
            event_groups[event_id].append(item)
            concept_sources[concept_id].add(source_id)
            concept_events[concept_id].add(event_id)
            source_events[source_id].add(event_id)

        concepts = sorted(concept_sources)
        event_revision_row = conn.execute(
            "SELECT value FROM graph_meta WHERE key='projection_revision'"
        ).fetchone()
        based_on_event_revision = int(event_revision_row[0]) if event_revision_row else 0
        conn.execute(
            """INSERT INTO field_revisions
               (revision,based_on_event_revision,status,metrics_json,created_at)
               VALUES(?,?,'PROPOSED',?,?)""",
            (
                revision,
                based_on_event_revision,
                encode({"reason": reason, "changed_source_id": changed_source_id}),
                now,
            ),
        )

        previous: dict[str, list[float]] = {}
        for row in conn.execute(
            """SELECT cp.cloud_id,cp.field_revision
               FROM semantic_cloud_current_projections cp
               WHERE cp.universe_id=?""",
            (FIELD_UNIVERSE,),
        ).fetchall():
            old = conn.execute(
                """SELECT applied_center_json
                   FROM semantic_cloud_projection_revisions
                   WHERE cloud_id=? AND universe_id=? AND field_revision=?""",
                (row["cloud_id"], FIELD_UNIVERSE, row["field_revision"]),
            ).fetchone()
            if old and old[0]:
                previous[str(row["cloud_id"])] = decode(old[0], [])

        cloud_ids = {concept: stable_id("semantic-cloud", concept) for concept in concepts}
        if concepts:
            placeholders = ",".join("?" for _ in concepts)
            conn.execute(
                f"DELETE FROM semantic_clouds WHERE id NOT IN ({placeholders})",
                [cloud_ids[concept] for concept in concepts],
            )
        else:
            conn.execute("DELETE FROM semantic_clouds")
            conn.execute("DELETE FROM contextual_cloud_projections")
            conn.execute(
                "UPDATE field_revisions SET status='VALIDATED',metrics_json=? WHERE revision=?",
                (encode({"updated_clouds": 0, "source_count": 0, "event_count": 0}), revision),
            )
            conn.execute(
                "UPDATE field_revisions SET status='APPLIED',applied_at=? WHERE revision=?",
                (now, revision),
            )
            return {
                "field_revision": revision,
                "updated_cloud_ids": [],
                "status": "APPLIED",
                "source_count": 0,
                "event_count": 0,
                "removed_source_id": changed_source_id,
            }

        # Build event-local coactivation.  Sharing a document source alone is
        # deliberately insufficient to create attraction.
        partner_events: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        pair_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
        pair_context: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for event_id, members in event_groups.items():
            unique = sorted({str(item["entity_id"]) for item in members})
            source_id = str(members[0]["source_id"])
            for concept in unique:
                for other in unique:
                    if other == concept:
                        continue
                    partner_events[concept][other].add(event_id)
                    pair_sources[(concept, other)].add(source_id)
                    pair_context[(concept, other)].extend(members)

        dimension_coordinates = self._dimension_coordinates(conn, concepts)
        bootstrap_centers: dict[str, list[float]] = {}
        display_bases: dict[str, list[float]] = {}
        for concept in concepts:
            cloud_id = cloud_ids[concept]
            existing = conn.execute(
                "SELECT bootstrap_center_json FROM semantic_clouds WHERE id=?",
                (cloud_id,),
            ).fetchone()
            bootstrap = decode(existing[0], []) if existing else _seed(concept)
            if len(bootstrap) != 3:
                bootstrap = _seed(concept)
            bootstrap_centers[concept] = [float(value) for value in bootstrap]

        for concept in concepts:
            bootstrap = bootstrap_centers[concept]
            partners = sorted(partner_events.get(concept, {}))
            if partners:
                partner_mean = [
                    sum(bootstrap_centers[other][index] for other in partners) / len(partners)
                    for index in range(3)
                ]
                # Preserve local identity while allowing event-local geometry.
                display = [
                    bootstrap[index] * 0.82 + partner_mean[index] * 0.18
                    for index in range(3)
                ]
                position_status = "LEARNED"
            else:
                display = list(bootstrap)
                position_status = "BOOTSTRAP"
            display_bases[concept] = [round(float(value), 6) for value in display]

            sources_for_concept = sorted(concept_sources[concept])
            event_count = len(concept_events[concept])
            independent_count = len(sources_for_concept)
            mass = min(8.0, 0.2 + 0.18 * independent_count + 0.04 * event_count)
            density = min(1.0, 0.2 + 0.08 * independent_count + 0.02 * event_count)
            stability = min(1.0, 0.18 + 0.09 * independent_count + 0.015 * event_count)
            active_dims = sorted(dimension_coordinates.get(concept, {}))
            conn.execute(
                """INSERT INTO semantic_clouds
                   (id,cloud_type,concept_id,mass,density,halo,stability,permeability,
                    bootstrap_center_json,learned_center_json,active_dimensions_json,
                    position_status,provenance_json,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(concept_id) DO UPDATE SET
                     mass=excluded.mass,density=excluded.density,halo=excluded.halo,
                     stability=excluded.stability,permeability=excluded.permeability,
                     bootstrap_center_json=excluded.bootstrap_center_json,
                     learned_center_json=excluded.learned_center_json,
                     active_dimensions_json=excluded.active_dimensions_json,
                     position_status=excluded.position_status,
                     provenance_json=excluded.provenance_json,
                     updated_at=excluded.updated_at""",
                (
                    cloud_ids[concept],
                    "concept",
                    concept,
                    mass,
                    density,
                    0.35,
                    stability,
                    0.5,
                    encode(bootstrap),
                    encode(display_bases[concept]),
                    encode(active_dims),
                    position_status,
                    encode(
                        [
                            {
                                "source_id": source_id,
                                "source_revision": self._source_revision(conn, source_id),
                                "event_ids": sorted(
                                    event_id
                                    for event_id in concept_events[concept]
                                    if event_id in source_events[source_id]
                                ),
                            }
                            for source_id in sources_for_concept
                        ]
                    ),
                    now,
                    now,
                ),
            )

        # Context projections are materialized from active evidence, not
        # incremented by technical rebuilds.
        conn.execute("DELETE FROM contextual_cloud_projections")
        context_counts: Counter[tuple[str, str]] = Counter()
        context_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for event_id, members in event_groups.items():
            for member in members:
                concept = str(member["entity_id"])
                cloud_id = cloud_ids[concept]
                context_key = f"predicate:{member['predicate_concept_id']}"
                context_counts[(cloud_id, context_key)] += 1
                context_rows[(cloud_id, context_key)].append(member)
        for (cloud_id, context_key), activation_count in sorted(context_counts.items()):
            concept = next(item for item, candidate in cloud_ids.items() if candidate == cloud_id)
            context_center = [
                round(value + seed * 0.015, 6)
                for value, seed in zip(display_bases[concept], _seed(context_key))
            ]
            context_id = stable_id("contextual-cloud", cloud_id, context_key)
            conn.execute(
                """INSERT INTO contextual_cloud_projections
                   (id,cloud_id,context_key,center_json,covariance_json,
                    activation_count,field_revision)
                   VALUES(?,?,?,?,?,?,?)""",
                (
                    context_id,
                    cloud_id,
                    context_key,
                    encode(context_center),
                    encode(_covariance(0.18)),
                    activation_count,
                    revision,
                ),
            )
            for member in context_rows[(cloud_id, context_key)]:
                source_id = str(member["source_id"])
                conn.execute(
                    """INSERT INTO contextual_cloud_projection_contributions
                       (source_id,source_revision,event_id,cloud_id,context_key,
                        activation_delta,field_revision,created_at)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        source_id,
                        self._source_revision(conn, source_id),
                        str(member["event_id"]),
                        cloud_id,
                        context_key,
                        1,
                        revision,
                        now,
                    ),
                )

        for concept, coordinates in dimension_coordinates.items():
            cloud_id = cloud_ids[concept]
            source_support = sorted(concept_sources[concept])
            confidence = min(1.0, 0.35 + 0.12 * len(source_support))
            for dimension_id, coordinate in sorted(coordinates.items()):
                conn.execute(
                    """INSERT INTO cloud_dimension_projections
                       (cloud_id,dimension_id,coordinate,variance,confidence,
                        source_support_json,revision)
                       VALUES(?,?,?,?,?,?,?)""",
                    (
                        cloud_id,
                        dimension_id,
                        coordinate,
                        0.1,
                        confidence,
                        encode(source_support),
                        revision,
                    ),
                )

        rejected_clouds: list[str] = []
        applied_clouds: list[str] = []
        for concept in concepts:
            cloud_id = cloud_ids[concept]
            base_center = display_bases[concept]
            cloud = conn.execute(
                "SELECT * FROM semantic_clouds WHERE id=?",
                (cloud_id,),
            ).fetchone()
            neighbours: list[dict[str, Any]] = []
            for other, shared_events in sorted(partner_events.get(concept, {}).items()):
                contexts = pair_context[(concept, other)]
                signatures = [decode(item.get("observation_signature_json"), {}) for item in contexts]
                prepositions = sorted(
                    {
                        str(item.get("preposition") or "")
                        for item in contexts
                        if item.get("preposition")
                    }
                )
                predicates = sorted({str(item["predicate_concept_id"]) for item in contexts})
                polarities = {str(item.get("polarity") or "POSITIVE").casefold() for item in contexts}
                relation_features = sorted(
                    {
                        str(signature.get("preposition") or signature.get("relation") or "")
                        for signature in signatures
                        if signature.get("preposition") or signature.get("relation")
                    }
                    | set(prepositions)
                )
                shared_sources = pair_sources[(concept, other)]
                weight = min(
                    1.0,
                    0.18 + 0.12 * len(shared_events) + 0.08 * len(shared_sources),
                )
                other_cloud = conn.execute(
                    "SELECT mass,density,halo FROM semantic_clouds WHERE id=?",
                    (cloud_ids[other],),
                ).fetchone()
                neighbours.append(
                    {
                        "cloud_id": cloud_ids[other],
                        "center": display_bases[other],
                        "learned_center": display_bases[other],
                        "mass": float(other_cloud["mass"]),
                        "density": float(other_cloud["density"]),
                        "halo": float(other_cloud["halo"]),
                        "coactivation_weight": weight,
                        "relation_attachment": relation_features,
                        "predicate_configuration": predicates,
                        "polarity": "negative" if "negative" in polarities else "positive",
                        "evidence_ids": sorted(shared_events),
                        "independent_source_keys": sorted(shared_sources),
                    }
                )

            measurements = dimension_coordinates.get(concept, {})
            dynamics = self.dynamics.calculate(
                {
                    "cloud_id": cloud_id,
                    "center": base_center,
                    "learned_center": base_center,
                    "mass": cloud["mass"],
                    "stability": cloud["stability"],
                    "density": cloud["density"],
                    "halo": cloud["halo"],
                },
                neighbours,
                active_measurements=measurements,
            )
            proposed = [
                round(base_center[index] + dynamics.limited_displacement[index], 6)
                for index in range(3)
            ]
            previous_revision_center = previous.get(cloud_id, [])
            finite = all(math.isfinite(value) for value in proposed)
            displacement = _distance(base_center, proposed)
            validation = {
                "passed": finite and displacement <= MAX_DISPLACEMENT + 1e-9,
                "finite": finite,
                "base_displacement": displacement,
                "maximum_displacement": MAX_DISPLACEMENT,
                "minimum_distance": MIN_DISTANCE,
                "neighbour_count": len(neighbours),
            }
            active_dims = sorted(measurements)
            principal_axes = [
                {
                    "dimension_id": dimension_id,
                    "coordinate": measurements[dimension_id],
                    "variance": 0.1,
                    "semantic_axis": True,
                }
                for dimension_id in active_dims
            ]
            if not principal_axes:
                principal_axes = [
                    {
                        "dimension_id": f"display:{axis}",
                        "axis": [1.0 if index == axis_index else 0.0 for index in range(3)],
                        "variance": 0.25,
                        "semantic_axis": False,
                    }
                    for axis_index, axis in enumerate(("x", "y", "z"))
                ]
            status = "VALIDATED" if validation["passed"] else "REJECTED"
            conn.execute(
                """INSERT INTO semantic_cloud_projection_revisions
                   (cloud_id,universe_id,field_revision,previous_center_json,
                    proposed_center_json,applied_center_json,covariance_json,
                    principal_axes_json,active_dimensions_json,validation_json,
                    status,created_at,applied_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    cloud_id,
                    FIELD_UNIVERSE,
                    revision,
                    encode(previous_revision_center),
                    encode(proposed),
                    encode(proposed) if validation["passed"] else None,
                    encode(_covariance()),
                    encode(principal_axes),
                    encode(active_dims),
                    encode(validation),
                    status,
                    now,
                    now if validation["passed"] else None,
                ),
            )
            transition = {
                "previous_revision_center": previous_revision_center,
                "reconstructed_base_center": base_center,
                "proposed_center": proposed,
                "applied_center": proposed if validation["passed"] else previous_revision_center,
                "force_components": [item.as_dict() for item in dynamics.forces],
                "supporting_event_ids": sorted(concept_events[concept]),
                "supporting_source_ids": sorted(concept_sources[concept]),
                "validation": validation,
                "max_displacement_applied": displacement >= MAX_DISPLACEMENT * 0.99,
            }
            if validation["passed"]:
                conn.execute(
                    """UPDATE semantic_cloud_projection_revisions
                       SET status='APPLIED' WHERE cloud_id=? AND universe_id=?
                       AND field_revision=?""",
                    (cloud_id, FIELD_UNIVERSE, revision),
                )
                conn.execute(
                    """INSERT OR REPLACE INTO semantic_cloud_projections
                       (cloud_id,universe_id,center_json,covariance_json,
                        principal_axes_json,local_potential,field_revision)
                       VALUES(?,?,?,?,?,?,?)""",
                    (
                        cloud_id,
                        FIELD_UNIVERSE,
                        encode(proposed),
                        encode(_covariance()),
                        encode(principal_axes),
                        -0.2,
                        revision,
                    ),
                )
                conn.execute(
                    """INSERT OR REPLACE INTO semantic_cloud_current_projections
                       (cloud_id,universe_id,field_revision) VALUES(?,?,?)""",
                    (cloud_id, FIELD_UNIVERSE, revision),
                )
                applied_clouds.append(cloud_id)
            else:
                rejected_clouds.append(cloud_id)
            conn.execute(
                """INSERT INTO field_transitions
                   (id,revision,source_id,cloud_id,transition_type,payload_json,created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (
                    stable_id("field-transition", revision, cloud_id),
                    revision,
                    changed_source_id or "rebuild",
                    cloud_id,
                    "DISPLACEMENT" if validation["passed"] else "REJECTED",
                    encode(transition),
                    now,
                ),
            )
            for force in dynamics.forces:
                conn.execute(
                    """INSERT INTO semantic_field_force_traces
                       (id,field_revision,cloud_id,force_type,source_cloud_id,
                        vector_json,magnitude,evidence_ids_json,payload_json,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        stable_id(
                            "field-force",
                            revision,
                            cloud_id,
                            force.type,
                            force.source_cloud_id or "self",
                        ),
                        revision,
                        cloud_id,
                        force.type,
                        force.source_cloud_id,
                        encode(list(force.vector)),
                        force.magnitude,
                        encode(list(force.evidence_ids)),
                        encode(force.payload or {}),
                        now,
                    ),
                )

        # Contribution history records source and field revisions separately.
        conn.execute(
            "DELETE FROM field_source_contributions WHERE field_revision=?",
            (revision,),
        )
        for source_id, event_ids in sorted(source_events.items()):
            source_revision = self._source_revision(conn, source_id)
            for event_id in sorted(event_ids):
                members = event_groups[event_id]
                for concept in sorted({str(item["entity_id"]) for item in members}):
                    cloud_id = cloud_ids[concept]
                    context_ids = sorted(
                        {
                            stable_id(
                                "contextual-cloud",
                                cloud_id,
                                f"predicate:{item['predicate_concept_id']}",
                            )
                            for item in members
                            if str(item["entity_id"]) == concept
                        }
                    )
                    force_delta = [
                        item.as_dict()
                        for item in self.dynamics.calculate(
                            {
                                "cloud_id": cloud_id,
                                "center": display_bases[concept],
                                "learned_center": display_bases[concept],
                                "mass": 1.0,
                                "stability": 0.0,
                                "density": 0.0,
                                "halo": 0.0,
                            },
                            [],
                        ).forces
                        if item.type not in {"MASS_INERTIA", "STABILITY_DAMPING"}
                    ]
                    conn.execute(
                        """INSERT INTO field_source_contributions
                           (source_id,source_revision,field_revision,event_id,cloud_id,
                            mass_delta,density_delta,force_delta_json,shape_delta_json,
                            context_projection_ids_json,created_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            source_id,
                            source_revision,
                            revision,
                            event_id,
                            cloud_id,
                            0.04,
                            0.03,
                            encode(force_delta),
                            encode({}),
                            encode(context_ids),
                            now,
                        ),
                    )

        metrics = {
            "updated_clouds": len(applied_clouds),
            "rejected_clouds": len(rejected_clouds),
            "source_count": len(source_events),
            "event_count": len(event_groups),
            "event_local_pair_count": sum(len(items) for items in partner_events.values()),
            "all_to_all_attraction": False,
        }
        final_status = "APPLIED" if applied_clouds or not concepts else "REJECTED"
        conn.execute(
            "UPDATE field_revisions SET status=?,metrics_json=?,applied_at=? WHERE revision=?",
            (final_status, encode(metrics), now if final_status == "APPLIED" else None, revision),
        )
        return {
            "field_revision": revision,
            "updated_cloud_ids": applied_clouds,
            "rejected_cloud_ids": rejected_clouds,
            "status": final_status,
            "source_count": len(source_events),
            "event_count": len(event_groups),
        }

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

    def snapshot(
        self,
        *,
        limit: int = 200,
        field_revision: int | None = None,
    ) -> dict[str, Any]:
        with self.repository.transaction() as conn:
            if field_revision is None:
                current = conn.execute(
                    """SELECT MAX(field_revision)
                       FROM semantic_cloud_current_projections
                       WHERE universe_id=?""",
                    (FIELD_UNIVERSE,),
                ).fetchone()
                field_revision = int(current[0]) if current and current[0] is not None else 0
            revision = conn.execute(
                """SELECT revision,based_on_event_revision,status,metrics_json,
                          created_at,applied_at
                   FROM field_revisions WHERE revision=?""",
                (field_revision,),
            ).fetchone()
            revision_number = int(revision["revision"]) if revision else 0
            rows = (
                conn.execute(
                    """SELECT c.*,p.previous_center_json,p.proposed_center_json,
                              p.applied_center_json AS center_json,p.covariance_json,
                              p.principal_axes_json,p.active_dimensions_json,
                              p.validation_json,p.status AS projection_status,
                              p.field_revision,e.canonical_lemma
                       FROM semantic_clouds c
                       JOIN semantic_cloud_projection_revisions p
                         ON p.cloud_id=c.id AND p.universe_id=?
                        AND p.field_revision=?
                       LEFT JOIN graph_entities e ON e.id=c.concept_id
                       ORDER BY c.stability DESC,c.mass DESC,c.id LIMIT ?""",
                    (
                        FIELD_UNIVERSE,
                        revision_number,
                        max(1, min(int(limit), 2000)),
                    ),
                ).fetchall()
                if revision_number
                else []
            )
            history = [
                {
                    "revision": int(item["revision"]),
                    "status": str(item["status"]),
                    "metrics": decode(item["metrics_json"], {}),
                    "created_at": str(item["created_at"]),
                    "applied_at": item["applied_at"],
                }
                for item in conn.execute(
                    """SELECT revision,status,metrics_json,created_at,applied_at
                       FROM field_revisions ORDER BY revision DESC LIMIT 20"""
                ).fetchall()
            ]
            contexts_by_cloud: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in conn.execute(
                """SELECT id,cloud_id,context_key,center_json,covariance_json,
                          activation_count,field_revision
                   FROM contextual_cloud_projections
                   ORDER BY cloud_id,activation_count DESC,context_key"""
            ).fetchall():
                contexts_by_cloud[str(item["cloud_id"])].append(
                    {
                        "id": str(item["id"]),
                        "context_key": str(item["context_key"]),
                        "center": decode(item["center_json"], []),
                        "covariance": decode(item["covariance_json"], []),
                        "activation_count": int(item["activation_count"]),
                        "field_revision": int(item["field_revision"]),
                    }
                )
            forces_by_cloud: dict[str, list[dict[str, Any]]] = defaultdict(list)
            if revision_number:
                for item in conn.execute(
                    """SELECT cloud_id,force_type,source_cloud_id,vector_json,
                              magnitude,evidence_ids_json,payload_json
                       FROM semantic_field_force_traces
                       WHERE field_revision=?
                       ORDER BY cloud_id,magnitude DESC,id""",
                    (revision_number,),
                ).fetchall():
                    forces_by_cloud[str(item["cloud_id"])].append(
                        {
                            "force_type": str(item["force_type"]),
                            "source_cloud_id": item["source_cloud_id"],
                            "vector": decode(item["vector_json"], []),
                            "magnitude": float(item["magnitude"]),
                            "evidence_ids": decode(item["evidence_ids_json"], []),
                            "payload": decode(item["payload_json"], {}),
                        }
                    )
            support_by_cloud: dict[str, dict[str, set[str]]] = defaultdict(
                lambda: {"sources": set(), "events": set()}
            )
            if revision_number:
                for item in conn.execute(
                    """SELECT cloud_id,source_id,event_id
                       FROM field_source_contributions
                       WHERE field_revision=?""",
                    (revision_number,),
                ).fetchall():
                    support_by_cloud[str(item["cloud_id"])]["sources"].add(
                        str(item["source_id"])
                    )
                    support_by_cloud[str(item["cloud_id"])]["events"].add(
                        str(item["event_id"])
                    )

        clouds = []
        for row in rows:
            cloud_id = str(row["id"])
            support = support_by_cloud[cloud_id]
            clouds.append(
                {
                    "cloud_id": cloud_id,
                    "cloud_type": str(row["cloud_type"]),
                    "concept_id": str(row["concept_id"]),
                    "canonical_lemma": str(row["canonical_lemma"] or row["concept_id"]),
                    "bootstrap_center": decode(row["bootstrap_center_json"], []),
                    "learned_center": decode(row["learned_center_json"], []),
                    "previous_revision_center": decode(row["previous_center_json"], []),
                    "proposed_center": decode(row["proposed_center_json"], []),
                    "center": decode(row["center_json"], []),
                    "display_center": decode(row["center_json"], []),
                    "position_status": str(row["position_status"]),
                    "projection_status": str(row["projection_status"]),
                    "validation": decode(row["validation_json"], {}),
                    "active_dimensions": decode(row["active_dimensions_json"], []),
                    "shape": {
                        "type": "ellipsoid",
                        "covariance": decode(row["covariance_json"], []),
                        "principal_axes": decode(row["principal_axes_json"], []),
                    },
                    "mass": float(row["mass"]),
                    "density": float(row["density"]),
                    "halo": float(row["halo"]),
                    "stability": float(row["stability"]),
                    "permeability": float(row["permeability"]),
                    "field_revision": int(row["field_revision"]),
                    "contextual_projections": contexts_by_cloud.get(cloud_id, []),
                    "supporting_source_ids": sorted(support["sources"]),
                    "supporting_event_ids": sorted(support["events"]),
                    "force_breakdown": forces_by_cloud.get(cloud_id, []),
                }
            )
        return {
            "universe_id": FIELD_UNIVERSE,
            "field_revision": revision_number,
            "revision": (
                {
                    "based_on_event_revision": int(revision["based_on_event_revision"]),
                    "status": str(revision["status"]),
                    "metrics": decode(revision["metrics_json"], {}),
                    "created_at": str(revision["created_at"]),
                    "applied_at": revision["applied_at"],
                }
                if revision
                else None
            ),
            "revision_history": history,
            "projection_method": "sparse_latent_semantics_with_3d_display_projection",
            "display_projection_warning": "Display projection — not semantic distance.",
            "clouds": clouds,
        }

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
