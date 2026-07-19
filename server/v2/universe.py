"""Self-organising, role-free micro-universes.

The graph remains the durable evidence layer.  This module derives a separate
geometric memory from observations: a stable entity is never conflated with a
concrete occurrence and discovered dimensions are represented as data rather
than application fields.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from server.core.settings import settings

from .acceleration import AccelerationRuntime, runtime as acceleration_runtime
from .graph_repository import GraphRepository, decode, encode, stable_id, utcnow


UNIVERSE_DEFINITIONS = (
    ("symbols", "Symbols", "symbol"),
    ("morphemes", "Fragments", "fragment"),
    ("word_forms", "Word Forms", "word_form"),
    ("words", "Words", "word"),
    ("usages", "Usages", "occurrence"),
    ("clauses", "Clauses", "clause"),
    ("events", "Events", "event"),
    ("scenes", "Scenes", "scene"),
    ("abstractions", "Abstractions", "abstraction"),
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _position(key: str) -> list[float]:
    """A stable initial base position; learning moves only via observations."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return [round((digest[index] / 255.0) * 2.0 - 1.0, 6) for index in range(3)]


def _similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    numerator = sum(float(left.get(key, 0.0)) * float(right.get(key, 0.0)) for key in keys)
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left.values()))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right.values()))
    return _clamp(numerator / (left_norm * right_norm)) if left_norm and right_norm else 0.0


def _mean_vectors(vectors: Iterable[Mapping[str, float]]) -> dict[str, float]:
    total: Counter[str] = Counter()
    count = 0
    for vector in vectors:
        total.update({str(key): float(value) for key, value in vector.items()})
        count += 1
    return {key: round(value / count, 6) for key, value in total.items()} if count else {}


@dataclass(frozen=True)
class UniverseFeatureAdapter:
    """Extracts observable context only; it has no semantic-role vocabulary."""

    universe_id: str

    def extract_features(
        self,
        tokens: Sequence[Mapping[str, Any]],
        index: int,
        morph: Mapping[str, Any],
    ) -> dict[str, float]:
        token = tokens[index]
        features: dict[str, float] = {
            f"shape:length:{min(len(str(token['normalized'])), 12)}": 0.35,
            f"position:sentence:{token['sentence_index']}": 0.25,
        }
        if index:
            features[f"context:left:{tokens[index - 1]['normalized']}"] = 1.0
        if index + 1 < len(tokens):
            features[f"context:right:{tokens[index + 1]['normalized']}"] = 1.0
        part = str(morph.get("part_of_speech") or "")
        if part:
            features[f"morph:part:{part}"] = 0.55
        for name, value in dict(morph.get("features") or {}).items():
            if value not in (None, "", False):
                features[f"morph:{name}:{value}"] = 0.3
        return features

    def build_context_vector(self, occurrence: Mapping[str, Any]) -> dict[str, float]:
        return {
            str(key): float(value)
            for key, value in dict(occurrence.get("observable_features") or {}).items()
        }

    def compare_contexts(self, left: Mapping[str, float], right: Mapping[str, float]) -> float:
        return _similarity(left, right)

    def estimate_prediction_gain(self, support: int, diversity: int) -> float:
        return _clamp((min(support, 8) / 8.0) * (0.45 + min(diversity, 4) / 8.0))


def _lexeme_features(observable: Mapping[str, float]) -> dict[str, float]:
    """Keep inflection and token position out of the persistent lexeme."""
    return {
        str(key): float(value)
        for key, value in observable.items()
        if not str(key).startswith(("morph:", "position:"))
    }


def _feature_family(feature: str) -> str:
    """Classify evidence without turning grammatical controls into meanings."""
    return (
        "semantic_structural"
        if feature.startswith(("context:left:", "context:right:", "structural:"))
        else "control"
    )


class DimensionDiscoverer:
    """Replaceable discovery contract used by every universe."""

    def collect_samples(self, conn: Any, universe_id: str) -> list[Mapping[str, Any]]:
        raise NotImplementedError

    def build_residuals(self, samples: Sequence[Mapping[str, Any]]) -> Mapping[str, list[Mapping[str, Any]]]:
        raise NotImplementedError

    def propose_candidates(self, conn: Any, universe_id: str) -> list[str]:
        raise NotImplementedError

    def evaluate_candidate(self, conn: Any, dimension_id: str) -> dict[str, float]:
        raise NotImplementedError

    def activate_candidate(self, conn: Any, dimension_id: str) -> str:
        raise NotImplementedError

    def update_existing(
        self,
        conn: Any,
        universe_id: str,
        source_id: str,
    ) -> int:
        return 0

    def merge_dimensions(self, conn: Any, universe_id: str) -> int:
        return 0

    def prune_dimensions(self, conn: Any, universe_id: str) -> int:
        return 0


class SparseResidualDiscoverer(DimensionDiscoverer):
    """First discoverer: sparse residual fields with evidence-based lifecycle."""

    max_candidate_dimensions = settings.max_candidate_dimensions_per_universe
    minimum_evidence = 2

    @staticmethod
    def _feature_value(sample: Mapping[str, Any], field: str) -> float:
        if field == "structural:source_cooccurrence":
            return 1.0
        return float(
            decode(sample["context_vector_json"], {}).get(field, 0.0)
        )

    @staticmethod
    def _domain_support(
        conn: Any,
        source_ids: set[str],
    ) -> int:
        if not source_ids:
            return 0
        rows = conn.execute(
            """SELECT source_type,metadata_json FROM knowledge_sources
               WHERE id IN ({})""".format(
                ",".join("?" for _ in source_ids)
            ),
            sorted(source_ids),
        ).fetchall()
        domains = {
            str(
                decode(row["metadata_json"], {}).get("domain_key")
                or row["source_type"]
            )
            for row in rows
        }
        return len(domains)

    @staticmethod
    def _split_support(
        conn: Any,
        source_ids: set[str],
    ) -> dict[str, int]:
        support = {"train": 0, "holdout": 0, "continual": 0}
        if not source_ids:
            return support
        rows = conn.execute(
            """SELECT independent_key FROM knowledge_sources
               WHERE id IN ({})""".format(
                ",".join("?" for _ in source_ids)
            ),
            sorted(source_ids),
        ).fetchall()
        for row in rows:
            key = str(row["independent_key"])
            for split in support:
                if f":{split}:" in key:
                    support[split] += 1
                    break
        return support

    def collect_samples(self, conn: Any, universe_id: str) -> list[Mapping[str, Any]]:
        return [
            dict(row)
            for row in conn.execute(
                """SELECT id,entity_id,source_id,context_id,context_vector_json
                   FROM universe_occurrences WHERE universe_id=?""",
                (universe_id,),
            ).fetchall()
        ]

    def build_residuals(self, samples: Sequence[Mapping[str, Any]]) -> Mapping[str, list[Mapping[str, Any]]]:
        fields: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for sample in samples:
            for key, value in decode(sample["context_vector_json"], {}).items():
                if float(value) > 0:
                    fields[str(key)].append(sample)
            # Source-local co-occurrence is structural evidence shared by
            # distinct entities; it is not a morphology or position control.
            fields["structural:source_cooccurrence"].append(sample)
        return fields

    @staticmethod
    def build_sparse_matrix(
        samples: Sequence[Mapping[str, Any]],
        runtime: AccelerationRuntime,
    ) -> tuple[Any, tuple[str, ...], Any]:
        """Return deterministic CSR/CSC semantic evidence without densifying.

        Control features remain available to morphology but deliberately never
        enter discovery.  The explicit size check also guards future callers
        from accidentally calling ``toarray`` on a large corpus.
        """
        semantic = sorted({
            str(key)
            for sample in samples
            for key, value in decode(sample["context_vector_json"], {}).items()
            if float(value) > 0 and _feature_family(str(key)) == "semantic_structural"
        } | ({"structural:source_cooccurrence"} if samples else set()))
        if not runtime.use("scipy"):
            return None, tuple(semantic), ()
        estimated_bytes = len(samples) * len(semantic) * 4
        if estimated_bytes > 64 * 1024 * 1024:
            runtime.fallback("scipy", "dense discovery estimate exceeds 64 MiB")
        positions = {name: index for index, name in enumerate(semantic)}
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for row_index, sample in enumerate(samples):
            vector = decode(sample["context_vector_json"], {})
            for name, value in vector.items():
                if name in positions and float(value) > 0:
                    rows.append(row_index)
                    columns.append(positions[name])
                    values.append(float(value))
            if "structural:source_cooccurrence" in positions:
                rows.append(row_index)
                columns.append(positions["structural:source_cooccurrence"])
                values.append(1.0)
        matrix = runtime.scipy.sparse.csr_matrix(
            (np.asarray(values, dtype=np.float32), (rows, columns)),
            shape=(len(samples), len(semantic)),
            dtype=np.float32,
        )
        return matrix, tuple(semantic), matrix.tocsc()

    @staticmethod
    def shadow_candidates(
        samples: Sequence[Mapping[str, Any]],
        runtime: AccelerationRuntime,
    ) -> list[dict[str, Any]]:
        """Produce non-admitting SVD/KMeans candidate metadata when available."""
        matrix, features, _ = SparseResidualDiscoverer.build_sparse_matrix(samples, runtime)
        if matrix is None or not runtime.use("sklearn"):
            return []
        if matrix.shape[0] < 3 or matrix.shape[1] < 2:
            return []
        components = min(8, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if components < 1:
            return []
        try:
            decomposition = runtime.sklearn.decomposition.TruncatedSVD(
                n_components=components, random_state=1729, n_iter=7,
            )
            reduced = decomposition.fit_transform(matrix)
            # Candidate clouds are capped and only annotate discovery; they
            # cannot change a dimension lifecycle or GraphMatcher admission.
            clusters = min(4, len(samples))
            labels = runtime.sklearn.cluster.MiniBatchKMeans(
                n_clusters=clusters, n_init=1, random_state=1729,
                reassignment_ratio=0,
            ).fit_predict(reduced)
        except Exception as exc:
            runtime.backend_failure("sklearn", exc)
            return []
        return [
            {"kind": "svd_kmeans", "component": index, "feature": features[index],
             "cluster": int(labels[index % len(labels)])}
            for index in range(min(components, len(features)))
        ]

    def propose_candidates(self, conn: Any, universe_id: str) -> list[str]:
        samples = self.collect_samples(conn, universe_id)
        # Native results are deliberately shadow-only: the existing field
        # discoverer remains the sole source that can advance a dimension.
        shadow_candidates = self.shadow_candidates(
            samples, getattr(self, "runtime", acceleration_runtime)
        )
        fields = self.build_residuals(samples)
        candidates = sorted(
            (
                (name, values)
                for name, values in fields.items()
                if len(values) >= self.minimum_evidence
                and _feature_family(name) == "semantic_structural"
            ),
            key=lambda item: (-len(item[1]), item[0]),
        )[: self.max_candidate_dimensions]
        now = utcnow()
        dimension_ids: list[str] = []
        for field, values in candidates:
            dimension_id = stable_id("dimension", universe_id, field)
            dimension_ids.append(dimension_id)
            entities = {str(value["entity_id"]) for value in values}
            sources = {str(value["source_id"]) for value in values}
            evidence = len(values)
            entity_support = len(entities)
            source_support = len(sources)
            domain_support = self._domain_support(conn, sources)
            split_support = self._split_support(conn, sources)
            diversity = min(entity_support, source_support)
            adapter = UniverseFeatureAdapter(universe_id)
            prediction = adapter.estimate_prediction_gain(evidence, diversity)
            stability = _clamp((
                min(1.0, entity_support / max(
                    1, settings.dimension_minimum_entity_support
                ))
                + min(1.0, source_support / max(
                    1, settings.dimension_minimum_source_support
                ))
            ) / 2.0)
            stability_lower_bound = _clamp(
                stability - 1.0 / math.sqrt(max(2, evidence))
            )
            strength = _clamp(sum(
                self._feature_value(value, field)
                for value in values
            ) / evidence)
            existing = conn.execute(
                """SELECT status,holdout_retrieval_gain
                   FROM latent_dimensions WHERE id=?""",
                (dimension_id,),
            ).fetchone()
            qualifies_for_probation = (
                entity_support >= settings.dimension_minimum_entity_support
                and source_support >= settings.dimension_minimum_source_support
                and domain_support >= settings.dimension_minimum_domain_support
            )
            can_activate = bool(
                qualifies_for_probation
                and stability >= settings.dimension_minimum_stability
                and stability_lower_bound
                >= settings.dimension_minimum_stability_lower_bound
                and existing is not None
                and float(existing["holdout_retrieval_gain"]) > 0
            )
            previous_status = str(existing["status"]) if existing else ""
            status = (
                previous_status
                if previous_status in {"shared", "frozen"} else
                "active" if can_activate else
                "probation" if qualifies_for_probation else
                "candidate"
            )
            canonical_id = stable_id(
                "canonical-dimension", universe_id, field
            )
            conn.execute(
                """INSERT INTO latent_dimensions
                   (id,canonical_dimension_id,revision,universe_id,
                    owner_scope,owner_id,representation_type,basis_json,
                    dimensionality,strength,stability,predictive_gain,retrieval_gain,
                    compression_gain,memory_cost,usage_count,
                    projection_usage_count,retrieval_contribution_count,
                    graph_admitted_contribution_count,
                    validated_answer_contribution_count,evidence_count,
                    entity_support,source_support,domain_support,
                    train_support,holdout_support,continual_support,
                    stability_lower_bound,status,created_at,last_updated_at,
                    last_confirmed_at)
                   VALUES(?,?,1,?, 'universe',NULL,'field',?,1,?,?,?,?,?,?,0,0,0,0,0,
                          ?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET strength=excluded.strength,
                     stability=excluded.stability,predictive_gain=excluded.predictive_gain,
                     retrieval_gain=excluded.retrieval_gain,
                     compression_gain=excluded.compression_gain,
                     evidence_count=excluded.evidence_count,
                     entity_support=excluded.entity_support,
                     source_support=excluded.source_support,
                     domain_support=excluded.domain_support,
                     train_support=excluded.train_support,
                     holdout_support=excluded.holdout_support,
                     continual_support=excluded.continual_support,
                     stability_lower_bound=excluded.stability_lower_bound,
                     status=excluded.status,last_updated_at=excluded.last_updated_at,
                     last_confirmed_at=excluded.last_confirmed_at""",
                (
                    dimension_id, canonical_id, universe_id, encode({
                        "residual_feature": field,
                        "feature_family": "semantic_structural",
                        "control_features_used": [],
                    }), strength,
                    stability, prediction, prediction, prediction * 0.5, 0.02,
                    evidence, entity_support, source_support, domain_support,
                    split_support["train"],
                    split_support["holdout"],
                    split_support["continual"],
                    stability_lower_bound, status, now, now, now,
                ),
            )
            conn.execute(
                """INSERT INTO dimension_lineage
                   (canonical_dimension_id,current_revision_id,
                    parent_dimension_ids_json,merged_from_json,split_from_json,
                    replaced_by,lineage_reason,updated_at)
                   VALUES(?,?,'[]','[]','[]',NULL,?,?)
                   ON CONFLICT(canonical_dimension_id) DO UPDATE SET
                     current_revision_id=excluded.current_revision_id,
                     lineage_reason=excluded.lineage_reason,
                     updated_at=excluded.updated_at""",
                (
                    canonical_id,
                    dimension_id,
                    "candidate refreshed from structural evidence",
                    now,
                ),
            )
            if existing is None:
                _record_event(conn, universe_id, "dimension_candidate_created", dimension_id, {"evidence": evidence})
            elif previous_status != status and status == "active":
                _record_event(conn, universe_id, "dimension_activated", dimension_id, {"evidence": evidence})
                conn.execute(
                    """UPDATE latent_dimensions
                       SET activated_at=COALESCE(activated_at,?)
                       WHERE id=?""",
                    (now, dimension_id),
                )
            if existing is None or previous_status != status:
                conn.execute(
                    """INSERT INTO dimension_history
                       (id,dimension_id,revision,status,snapshot_json,reason,created_at)
                       VALUES(?,?,?,?,?,?,?)""",
                    (
                        stable_id(
                            "dimension-history", dimension_id, status, now
                        ),
                        dimension_id,
                        1,
                        status,
                        encode({
                            "entity_support": entity_support,
                            "source_support": source_support,
                            "domain_support": domain_support,
                            "stability": stability,
                            "stability_lower_bound": stability_lower_bound,
                        }),
                        "structural discovery evaluation",
                        now,
                    ),
                )
            self._write_projections(conn, dimension_id, field, values, now)
            self._write_dimension_cloud(conn, dimension_id, field, values, stability, now)
        self._write_relations(conn, universe_id, candidates)
        # Library-generated components remain *candidate* metadata.  They
        # have neither projections nor lifecycle authority, so they cannot
        # bypass support/stability/holdout checks or affect GraphMatcher.
        for shadow in shadow_candidates:
            feature = "sklearn:{kind}:{component}:{cluster}:{feature}".format(**shadow)
            dimension_id = stable_id("dimension", universe_id, feature)
            dimension_ids.append(dimension_id)
            conn.execute(
                """INSERT INTO latent_dimensions
                   (id,canonical_dimension_id,universe_id,owner_scope,
                    representation_type,basis_json,status,created_at,last_updated_at)
                   VALUES(?,?,?,'universe','subspace',?,'candidate',?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     basis_json=excluded.basis_json,status='candidate',
                     last_updated_at=excluded.last_updated_at""",
                (
                    dimension_id,
                    stable_id("canonical-dimension", universe_id, feature),
                    universe_id,
                    encode({
                        "backend": "sklearn",
                        "shadow": True,
                        "feature_family": "semantic_structural",
                        "residual_feature": feature,
                        **shadow,
                    }),
                    now,
                    now,
                ),
            )
        # Apply the same configured budget after every candidate generator
        # has contributed.  Running this before the shadow generator would
        # allow native candidates to exceed the per-universe cap.
        self._enforce_capacity(conn, universe_id)
        return dimension_ids

    def update_existing(
        self,
        conn: Any,
        universe_id: str,
        source_id: str,
    ) -> int:
        """Project one source without re-running global candidate discovery."""
        samples = [
            dict(row)
            for row in conn.execute(
                """SELECT id,entity_id,source_id,context_id,
                          context_vector_json
                   FROM universe_occurrences
                   WHERE universe_id=? AND source_id=?""",
                (universe_id, source_id),
            ).fetchall()
        ]
        if not samples:
            return 0
        dimensions = conn.execute(
            """SELECT id,basis_json FROM latent_dimensions
               WHERE universe_id=? AND status NOT IN ('merged','pruned')""",
            (universe_id,),
        ).fetchall()
        now = utcnow()
        updated = 0
        for dimension in dimensions:
            field = str(
                decode(dimension["basis_json"], {}).get(
                    "residual_feature"
                ) or ""
            )
            matching = [
                sample for sample in samples
                if self._feature_value(sample, field) > 0
            ]
            if not matching:
                continue
            self._write_projections(
                conn,
                str(dimension["id"]),
                field,
                matching,
                now,
            )
            updated += 1
        return updated

    def _write_relations(
        self,
        conn: Any,
        universe_id: str,
        candidates: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    ) -> None:
        """Relate co-supported fields; no field is assumed to contain another."""
        for index, (left_field, left_values) in enumerate(candidates):
            left_by_id = {
                str(value["id"]): value for value in left_values
            }
            left_ids = set(left_by_id)
            for right_field, right_values in candidates[index + 1:]:
                right_by_id = {
                    str(value["id"]): value for value in right_values
                }
                right_ids = set(right_by_id)
                overlap = left_ids & right_ids
                if len(overlap) < self.minimum_evidence:
                    continue
                overlap_sources = {
                    str(left_by_id[item]["source_id"])
                    for item in overlap
                } | {
                    str(right_by_id[item]["source_id"])
                    for item in overlap
                }
                # A source-local coincidence is not evidence for a stable
                # cross-source dimension relation.
                if len(overlap_sources) < 2:
                    continue
                weight = _clamp(len(overlap) / max(len(left_ids), len(right_ids)))
                left_id = stable_id("dimension", universe_id, left_field)
                right_id = stable_id("dimension", universe_id, right_field)
                relation_type = (
                    "merged_from" if weight >= 0.98 else
                    "overlapping" if weight >= 0.30 else
                    "correlated"
                )
                conn.execute(
                    """INSERT INTO dimension_relations
                       (source_dimension_id,target_dimension_id,relation_type,weight,confidence,evidence_count)
                       VALUES(?,?,?,?,?,?)
                       ON CONFLICT(source_dimension_id,target_dimension_id,relation_type) DO UPDATE SET
                         weight=excluded.weight,confidence=excluded.confidence,evidence_count=excluded.evidence_count""",
                    (
                        left_id,
                        right_id,
                        relation_type,
                        weight,
                        weight,
                        len(overlap),
                    ),
                )
                if relation_type == "merged_from":
                    conn.execute(
                        """UPDATE latent_dimensions SET status='merged',
                           last_updated_at=? WHERE id=? AND status IN
                           ('candidate','probation','weak')""",
                        (utcnow(), right_id),
                    )
                    _record_event(
                        conn,
                        universe_id,
                        "dimension_merged",
                        right_id,
                        {"merged_into": left_id, "overlap_score": weight},
                    )

    @staticmethod
    def _enforce_capacity(conn: Any, universe_id: str) -> None:
        limits = {
            "candidate": settings.max_candidate_dimensions_per_universe,
            "probation": settings.max_probation_dimensions_per_universe,
            "active": settings.max_active_dimensions_per_universe,
        }
        for status, limit in limits.items():
            rows = conn.execute(
                """SELECT id FROM latent_dimensions
                   WHERE universe_id=? AND status=?
                   ORDER BY stability DESC,retrieval_gain DESC,
                            evidence_count DESC,id""",
                (universe_id, status),
            ).fetchall()
            for row in rows[max(0, limit):]:
                conn.execute(
                    """UPDATE latent_dimensions SET status='weak',
                       last_updated_at=? WHERE id=?""",
                    (utcnow(), row["id"]),
                )
                _record_event(
                    conn,
                    universe_id,
                    "dimension_pruned",
                    str(row["id"]),
                    {
                        "reason": "capacity_limit",
                        "previous_status": status,
                        "configured_limit": limit,
                    },
                )

    def _write_projections(
        self, conn: Any, dimension_id: str, field: str, values: Sequence[Mapping[str, Any]], now: str
    ) -> None:
        entity_values: dict[str, list[float]] = defaultdict(list)
        occurrence_rows = []
        for sample in values:
            membership = _clamp(self._feature_value(sample, field))
            entity_values[str(sample["entity_id"])].append(membership)
            occurrence_rows.append((
                stable_id("projection", dimension_id, "occurrence", sample["id"]), dimension_id,
                sample["id"], sample["context_id"], encode([membership]), membership,
                1.0 - membership, 0.75, now,
            ))
        if occurrence_rows:
            conn.executemany(
                """INSERT INTO projections
                   (id,dimension_id,source_type,source_id,context_id,coordinates_json,
                    membership,distance_to_core,confidence,calculated_at)
                   VALUES(?,?, 'occurrence',?,?,?, ?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     coordinates_json=excluded.coordinates_json,membership=excluded.membership,
                     distance_to_core=excluded.distance_to_core,calculated_at=excluded.calculated_at""",
                occurrence_rows,
            )
        entity_rows = []
        for entity_id, memberships in entity_values.items():
            membership = sum(memberships) / len(memberships)
            entity_rows.append((
                stable_id("projection", dimension_id, "entity", entity_id), dimension_id,
                entity_id, encode([round(membership, 6)]), membership,
                1.0 - membership, 0.8, now,
            ))
        if entity_rows:
            conn.executemany(
                """INSERT INTO projections
                   (id,dimension_id,source_type,source_id,context_id,coordinates_json,
                    membership,distance_to_core,confidence,calculated_at)
                   VALUES(?,?, 'entity',?,'',?, ?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     coordinates_json=excluded.coordinates_json,membership=excluded.membership,
                     distance_to_core=excluded.distance_to_core,calculated_at=excluded.calculated_at""",
                entity_rows,
            )

    def _write_dimension_cloud(
        self, conn: Any, dimension_id: str, field: str,
        values: Sequence[Mapping[str, Any]], stability: float, now: str,
    ) -> None:
        memberships = [
            self._feature_value(item, field)
            for item in values
        ]
        density = sum(memberships) / max(1, len(memberships))
        conn.execute(
            """INSERT INTO dimension_clouds
               (id,dimension_id,core_vector_json,boundary_region_json,
                applicability_region_json,mass,gravity,radius,density,stability)
               VALUES(?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET core_vector_json=excluded.core_vector_json,
                 boundary_region_json=excluded.boundary_region_json,
                 applicability_region_json=excluded.applicability_region_json,mass=excluded.mass,
                 density=excluded.density,stability=excluded.stability""",
            (stable_id("dimension-cloud", dimension_id), dimension_id,
             encode({"residual_feature": field}), encode({"membership": [0.2, 0.6]}),
             encode({"observed": len(values)}), len(values), density, 1.0 - density,
             density, stability),
        )

    def evaluate_candidate(self, conn: Any, dimension_id: str) -> dict[str, float]:
        row = conn.execute(
            """SELECT predictive_gain,retrieval_gain,compression_gain,memory_cost,
                      stability,evidence_count FROM latent_dimensions WHERE id=?""",
            (dimension_id,),
        ).fetchone()
        if row is None:
            raise KeyError(dimension_id)
        utility = _clamp(float(row["predictive_gain"]) + float(row["retrieval_gain"])
                         + float(row["compression_gain"]) - float(row["memory_cost"]))
        return {"utility": utility, "stability": float(row["stability"]), "evidence": float(row["evidence_count"])}

    def activate_candidate(self, conn: Any, dimension_id: str) -> str:
        row = conn.execute(
            """SELECT stability,stability_lower_bound,holdout_retrieval_gain,
                      entity_support,source_support,domain_support
               FROM latent_dimensions WHERE id=?""",
            (dimension_id,),
        ).fetchone()
        if row is None:
            raise KeyError(dimension_id)
        status = "active" if (
            float(row["stability"]) >= settings.dimension_minimum_stability
            and float(row["stability_lower_bound"])
            >= settings.dimension_minimum_stability_lower_bound
            and float(row["holdout_retrieval_gain"]) > 0
            and int(row["entity_support"])
            >= settings.dimension_minimum_entity_support
            and int(row["source_support"])
            >= settings.dimension_minimum_source_support
            and int(row["domain_support"])
            >= settings.dimension_minimum_domain_support
        ) else "probation"
        now = utcnow()
        conn.execute(
            """UPDATE latent_dimensions SET status=?,
               activated_at=CASE WHEN ?='active'
                 THEN COALESCE(activated_at,?) ELSE activated_at END,
               last_updated_at=? WHERE id=?""",
            (status, status, now, now, dimension_id),
        )
        return status

    def prune_dimensions(self, conn: Any, universe_id: str) -> int:
        """Retire fields that no longer have enough surviving evidence."""
        rows = conn.execute(
            "SELECT id,basis_json FROM latent_dimensions WHERE universe_id=? AND status<>'pruned'",
            (universe_id,),
        ).fetchall()
        samples = self.collect_samples(conn, universe_id)
        pruned = 0
        for row in rows:
            feature = str(decode(row["basis_json"], {}).get("residual_feature") or "")
            evidence = sum(
                float(decode(sample["context_vector_json"], {}).get(feature, 0.0)) > 0
                for sample in samples
            )
            if evidence < self.minimum_evidence:
                conn.execute("UPDATE latent_dimensions SET status='pruned',evidence_count=? WHERE id=?", (evidence, row["id"]))
                _record_event(conn, universe_id, "dimension_pruned", str(row["id"]), {"evidence": evidence})
                pruned += 1
        return pruned


def _record_event(conn: Any, universe_id: str, event_type: str, subject_id: str, payload: Mapping[str, Any]) -> None:
    conn.execute(
        """INSERT INTO universe_training_events(id,universe_id,event_type,subject_id,payload_json,created_at)
           VALUES(?,?,?,?,?,?)""",
        (stable_id("universe-history", universe_id, event_type, subject_id, utcnow()), universe_id,
         event_type, subject_id, encode(dict(payload)), utcnow()),
    )


class UniverseService:
    """Persistence, discovery and query facade for all dynamic universes."""

    def __init__(self, repository: Optional[GraphRepository] = None, *, runtime: Optional[AccelerationRuntime] = None) -> None:
        self.repository = repository or GraphRepository()
        self.acceleration = runtime or acceleration_runtime
        self.discoverer: DimensionDiscoverer = SparseResidualDiscoverer()
        # The discoverer is intentionally stateless except for this optional
        # runtime, so tests and experiments can inject an isolated instance.
        self.discoverer.runtime = self.acceleration  # type: ignore[attr-defined]
        self._ensure_universes()

    def _ensure_universes(self) -> None:
        now = utcnow()
        with self.repository.transaction() as conn:
            for universe_id, name, scale in UNIVERSE_DEFINITIONS:
                conn.execute(
                    """INSERT INTO universes
                       (id,name,scale,version,base_space_config_json,discovery_config_json,
                        statistics_json,created_at,updated_at)
                       VALUES(?,?,?,?,?,?, '{}',?,?) ON CONFLICT(id) DO NOTHING""",
                    (universe_id, name, scale, "2.8.0",
                     encode({"dimensions": 3, "metric": "cosine+local"}),
                     encode({"algorithm": "sparse_residual", "max_active_dimensions": 32,
                             "minimum_evidence": 2}), now, now),
                )

    def reset(self) -> dict[str, Any]:
        self.repository.reset()
        self._ensure_universes()
        return {"reset": True, "universes": self.list_universes()["universes"]}

    def export_memory(self) -> dict[str, Any]:
        """Return a portable JSON snapshot of every current-memory table."""
        with self.repository.transaction() as conn:
            tables = [
                str(row["name"])
                for row in conn.execute(
                    """SELECT name FROM sqlite_master
                       WHERE type='table' AND name NOT LIKE 'sqlite_%'
                       ORDER BY name"""
                ).fetchall()
            ]
            payload: dict[str, list[dict[str, Any]]] = {}
            for table in tables:
                rows = conn.execute(f'SELECT * FROM "{table.replace(chr(34), chr(34) * 2)}"').fetchall()
                payload[table] = [self._export_row(row) for row in rows]
            meta = {
                str(row["key"]): str(row["value"])
                for row in conn.execute("SELECT key,value FROM graph_meta ORDER BY key").fetchall()
            }
        return {
            "format": "superai-memory-export",
            "exported_at": utcnow(),
            "schema": meta,
            "universes": self.list_universes()["universes"],
            "tables": payload,
        }

    def ingest_source(
        self,
        source_id: str,
        *,
        discover_dimensions: bool = True,
        connection: Any = None,
    ) -> dict[str, Any]:
        """Materialise confirmed graph evidence into the generic universe model."""
        touched: set[str] = set()
        transaction = (
            nullcontext(connection)
            if connection is not None
            else self.repository.transaction()
        )
        with transaction as conn:
            observed_at = utcnow()
            training_event_rows: list[tuple[Any, ...]] = []
            transition_counts: Counter[tuple[str, ...]] = Counter()
            entity_states: dict[
                str, tuple[int, Mapping[str, float]]
            ] = {}

            def queue_transition(
                source_universe: str,
                target_universe: str,
                source_type: str,
                source_value_id: str,
                target_type: str,
                target_value_id: str,
            ) -> None:
                transition_counts[(
                    source_universe,
                    target_universe,
                    source_type,
                    source_value_id,
                    target_type,
                    target_value_id,
                    source_id,
                )] += 1

            source = conn.execute("SELECT status FROM knowledge_sources WHERE id=?", (source_id,)).fetchone()
            if source is None:
                raise KeyError(source_id)
            if str(source["status"]) != "CONFIRMED":
                return {"source_id": source_id, "ingested": False, "reason": "not_confirmed"}
            token_rows = [dict(row) for row in conn.execute(
                "SELECT id,token_index,sentence_index,surface,normalized,selected_hypothesis_id FROM graph_tokens WHERE source_id=? ORDER BY token_index",
                (source_id,),
            ).fetchall()]
            morph_by_token = {
                str(row["token_id"]): dict(row)
                for row in conn.execute(
                    """SELECT token_id,lemma,part_of_speech,morph_score,features_json
                       FROM graph_morph_hypotheses
                       WHERE selected=1 AND token_id IN (SELECT id FROM graph_tokens WHERE source_id=?)""",
                    (source_id,),
                ).fetchall()
            }
            word_usages: list[tuple[str, str, str]] = []
            adapter = UniverseFeatureAdapter("words")
            for index, token in enumerate(token_rows):
                morph = morph_by_token.get(str(token["id"]), {})
                morph = {**morph, "features": decode(morph.get("features_json"), {})}
                observable = adapter.extract_features(token_rows, index, morph)
                language = "ru"
                lemma = str(morph.get("lemma") or token["normalized"]).casefold()
                normalized_surface = str(token["normalized"]).casefold()
                proper_name = bool(morph["features"].get("proper_name"))
                display_surface = (
                    str(token["surface"])
                    if proper_name else normalized_surface
                )
                # The final component is deliberately a sense slot.  It is
                # empty in the first release, but makes a future лук#1/лук#2
                # split an additive operation rather than an identity change.
                sense_cluster_id = ""
                lexeme_observable = _lexeme_features(observable)
                # Script is a lexical property rather than an inflectional or
                # occurrence-local one; it also gives the discoverer a safe
                # cross-token signal before richer contexts accumulate.
                lexeme_observable[
                    "shape:script:cyrillic"
                    if any("а" <= char <= "я" or char == "ё" for char in lemma)
                    else "shape:script:other"
                ] = 0.2
                token_index = int(token["token_index"])
                lexeme_key = f"{language}:{lemma}:{sense_cluster_id}"
                lexeme_entity_id = stable_id(
                    "universe-entity", "words", lexeme_key
                )
                lexeme_occurrence_id = stable_id(
                    "universe-occurrence",
                    "words",
                    source_id,
                    f"lexeme:{token_index}",
                    lexeme_entity_id,
                )
                form_key = f"{language}:{lemma}:{display_surface}"
                form_entity_id = stable_id(
                    "universe-entity", "word_forms", form_key
                )
                form_occurrence_id = stable_id(
                    "universe-occurrence",
                    "word_forms",
                    source_id,
                    f"word-form:{token_index}",
                    form_entity_id,
                )
                (
                    (lexeme_entity, lexeme_occurrence),
                    (word_form_entity, word_form_occurrence),
                    (usage_entity, usage_occurrence),
                ) = self._observe_many(
                    conn,
                    None,
                    source_id,
                    [
                        {
                            "universe_id": "words",
                            "key": lexeme_key,
                            "display": lemma,
                            "context_id": f"lexeme:{token_index}",
                            "observable": lexeme_observable,
                            "confidence": float(
                                morph.get("morph_score") or 0.8
                            ),
                        },
                        {
                            "universe_id": "word_forms",
                            "key": form_key,
                            "display": display_surface,
                            "context_id": f"word-form:{token_index}",
                            "observable": observable,
                            "confidence": float(
                                morph.get("morph_score") or 0.8
                            ),
                            "parent_occurrence_id": lexeme_occurrence_id,
                        },
                        {
                            "universe_id": "usages",
                            "key": f"{source_id}:{token_index}",
                            "display": str(token["surface"]),
                            "context_id": f"usage:{token_index}",
                            "observable": observable,
                            "confidence": 0.75,
                            "parent_occurrence_id": form_occurrence_id,
                        },
                    ],
                    observed_at=observed_at,
                    event_rows=training_event_rows,
                    entity_states=entity_states,
                )
                self._link_lexeme(
                    conn,
                    lexeme_entity_id=lexeme_entity,
                    language=language,
                    canonical_lemma=lemma,
                    sense_cluster_id=sense_cluster_id,
                    observed_at=observed_at,
                )
                touched.add("words")
                self._link_word_form(
                    conn,
                    word_form_entity_id=word_form_entity,
                    lexeme_entity_id=lexeme_entity,
                    language=language,
                    normalized_surface=display_surface,
                    display_surface=display_surface,
                    morphology_features=morph["features"],
                    morphology_confidence=float(morph.get("morph_score") or 0.8),
                    observed_at=observed_at,
                )
                queue_transition(
                    "word_forms", "words", "entity", word_form_entity,
                    "entity", lexeme_entity,
                )
                touched.add("word_forms")
                self._link_word_usage(
                    conn,
                    usage_occurrence_id=usage_occurrence,
                    lexeme_entity_id=lexeme_entity,
                    word_form_entity_id=word_form_entity,
                    source_id=source_id,
                    sentence_index=int(token["sentence_index"]),
                    token_index=token_index,
                    observed_at=observed_at,
                )
                word_usages.append((lexeme_entity, word_form_entity, usage_occurrence))
                queue_transition(
                    "words", "usages", "entity", lexeme_entity,
                    "occurrence", usage_occurrence,
                )
                queue_transition(
                    "word_forms", "usages", "entity", word_form_entity,
                    "occurrence", usage_occurrence,
                )
                touched.add("usages")
                symbol_observations = [
                    {
                        "key": character,
                        "display": character,
                        "context_id": (
                            f"token:{token['token_index']}:char:{char_index}"
                        ),
                        "observable": {
                            "position:character": (
                                min(char_index, 12) / 12.0
                            )
                        },
                        "confidence": 0.7,
                    }
                    for char_index, character in enumerate(normalized_surface)
                ]
                for symbol_entity, _ in self._observe_many(
                    conn,
                    "symbols",
                    source_id,
                    symbol_observations,
                    observed_at=observed_at,
                    event_rows=training_event_rows,
                    entity_states=entity_states,
                ):
                    queue_transition(
                        "symbols", "word_forms", "entity", symbol_entity,
                        "entity", word_form_entity,
                    )
                    touched.add("symbols")
                fragment_observations = []
                for size in range(2, min(4, len(normalized_surface)) + 1):
                    for start in range(len(normalized_surface) - size + 1):
                        fragment = normalized_surface[start:start + size]
                        fragment_observations.append({
                            "key": fragment,
                            "display": fragment,
                            "context_id": (
                                f"token:{token['token_index']}:"
                                f"fragment:{start}:{size}"
                            ),
                            "observable": {
                                "shape:length": float(size) / 4.0,
                                "position:fragment": (
                                    float(start)
                                    / max(1, len(normalized_surface))
                                ),
                            },
                            "confidence": 0.55,
                        })
                for fragment_entity, _ in self._observe_many(
                    conn,
                    "morphemes",
                    source_id,
                    fragment_observations,
                    observed_at=observed_at,
                    event_rows=training_event_rows,
                    entity_states=entity_states,
                ):
                        queue_transition(
                            "morphemes", "word_forms", "entity",
                            fragment_entity, "entity", word_form_entity,
                        )
                        touched.add("morphemes")
            for sentence, rows in self._group_sentences(token_rows).items():
                key = " ".join(str(row["normalized"]) for row in rows)
                clause_entity, clause_occurrence = self._observe(
                    conn, "clauses", key, " ".join(str(row["surface"]) for row in rows), source_id,
                    f"sentence:{sentence}", {"shape:token_count": min(len(rows), 12) / 12.0}, confidence=0.65,
                    observed_at=observed_at,
                    event_rows=training_event_rows,
                    entity_states=entity_states,
                )
                touched.add("clauses")
                for _, _, usage_occurrence in word_usages:
                    queue_transition(
                        "usages", "clauses", "occurrence",
                        usage_occurrence, "occurrence", clause_occurrence,
                    )
            event_rows = conn.execute(
                "SELECT id,predicate_lemma,predicate_surface,confidence FROM graph_events WHERE source_id=?", (source_id,)
            ).fetchall()
            for event in event_rows:
                event_entity, event_occurrence = self._observe(
                    conn, "events", str(event["predicate_lemma"]), str(event["predicate_surface"]), source_id,
                    f"event:{event['id']}", {"context:predicate": 1.0}, confidence=float(event["confidence"]),
                    observed_at=observed_at,
                    event_rows=training_event_rows,
                    entity_states=entity_states,
                )
                touched.add("events")
                for _, _, usage_occurrence in word_usages:
                    queue_transition(
                        "usages", "events", "occurrence",
                        usage_occurrence, "occurrence", event_occurrence,
                    )
            scene_entity, scene_occurrence = self._observe(
                conn, "scenes", source_id, source_id, source_id, "source", {"shape:source": 1.0}, confidence=0.65,
                observed_at=observed_at,
                event_rows=training_event_rows,
                entity_states=entity_states,
            )
            touched.add("scenes")
            for event in event_rows:
                event_entity = stable_id("universe-entity", "events", str(event["predicate_lemma"]))
                queue_transition(
                    "events", "scenes", "entity", event_entity,
                    "occurrence", scene_occurrence,
                )
            for universe_id in touched:
                if discover_dimensions:
                    self.discoverer.propose_candidates(conn, universe_id)
                else:
                    self.discoverer.update_existing(
                        conn, universe_id, source_id
                    )
                if discover_dimensions:
                    self._refresh_clouds(conn, universe_id)
                self._refresh_statistics(conn, universe_id)
            active_dimensions = conn.execute(
                """SELECT id,universe_id,basis_json FROM latent_dimensions
                   WHERE universe_id IN ({}) AND status IN ('active','shared')""".format(
                    ",".join("?" for _ in touched)
                ),
                sorted(touched),
            ).fetchall() if touched else []
            for dimension in active_dimensions:
                basis = decode(dimension["basis_json"], {})
                abstraction_id, _ = self._observe(
                    conn, "abstractions", str(dimension["id"]), str(dimension["id"]), source_id,
                    f"dimension:{dimension['id']}",
                    {f"field:{basis.get('residual_feature', dimension['id'])}": 1.0},
                    confidence=0.65,
                    observed_at=observed_at,
                    event_rows=training_event_rows,
                    entity_states=entity_states,
                )
                queue_transition(
                    str(dimension["universe_id"]), "abstractions",
                    "dimension", str(dimension["id"]), "entity",
                    abstraction_id,
                )
                touched.add("abstractions")
            if "abstractions" in touched:
                if discover_dimensions:
                    self.discoverer.propose_candidates(
                        conn, "abstractions"
                    )
                else:
                    self.discoverer.update_existing(
                        conn, "abstractions", source_id
                    )
                if discover_dimensions:
                    self._refresh_clouds(conn, "abstractions")
                self._refresh_statistics(conn, "abstractions")
            self._write_transitions(
                conn,
                transition_counts,
                observed_at=observed_at,
                event_rows=training_event_rows,
            )
            if training_event_rows:
                conn.executemany(
                    """INSERT INTO universe_training_events
                       (id,universe_id,event_type,subject_id,payload_json,
                        created_at)
                       VALUES(?,?,?,?,?,?)""",
                    training_event_rows,
                )
            # A source transaction changes both projection memberships and
            # vertical edges.  Increment each revision exactly once only
            # after all materialisation work has succeeded.
            self.repository.bump_revisions(conn)
        return {"source_id": source_id, "ingested": True, "universes": sorted(touched)}

    def remove_source(self, source_id: str) -> dict[str, Any]:
        """Forget derived geometry together with retracted source evidence."""
        with self.repository.transaction() as conn:
            affected = [str(row[0]) for row in conn.execute(
                "SELECT DISTINCT universe_id FROM universe_occurrences WHERE source_id=?", (source_id,)
            ).fetchall()]
            if not affected:
                return {"source_id": source_id, "removed": False, "universes": []}
            conn.execute("DELETE FROM universe_occurrences WHERE source_id=?", (source_id,))
            conn.execute("DELETE FROM universe_transitions WHERE context_id=?", (source_id,))
            for universe_id in affected:
                self.discoverer.prune_dimensions(conn, universe_id)
                conn.execute("DELETE FROM entity_clouds WHERE universe_id=?", (universe_id,))
                conn.execute("DELETE FROM universe_entities WHERE universe_id=? AND id NOT IN (SELECT DISTINCT entity_id FROM universe_occurrences WHERE universe_id=?)", (universe_id, universe_id))
                conn.execute("UPDATE universe_entities SET frequency=(SELECT COUNT(*) FROM universe_occurrences o WHERE o.entity_id=universe_entities.id),updated_at=? WHERE universe_id=?", (utcnow(), universe_id))
                self._refresh_clouds(conn, universe_id)
                self._refresh_statistics(conn, universe_id)
            self.repository.bump_revisions(conn)
        return {"source_id": source_id, "removed": True, "universes": affected}

    @staticmethod
    def _group_sentences(tokens: Sequence[Mapping[str, Any]]) -> Mapping[int, list[Mapping[str, Any]]]:
        grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for token in tokens:
            grouped[int(token["sentence_index"])].append(token)
        return grouped

    @staticmethod
    def _link_word_form(
        conn: Any,
        *,
        word_form_entity_id: str,
        lexeme_entity_id: str,
        language: str,
        normalized_surface: str,
        display_surface: str,
        morphology_features: Mapping[str, Any],
        morphology_confidence: float,
        observed_at: Optional[str] = None,
    ) -> None:
        now = observed_at or utcnow()
        conn.execute(
            """INSERT INTO word_forms
               (word_form_entity_id,lexeme_entity_id,language,normalized_surface,
                display_surface,morphological_features_json,morphology_confidence,
                created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(word_form_entity_id) DO UPDATE SET
                 morphological_features_json=excluded.morphological_features_json,
                 morphology_confidence=MAX(word_forms.morphology_confidence,
                                           excluded.morphology_confidence),
                 updated_at=excluded.updated_at""",
            (
                word_form_entity_id,
                lexeme_entity_id,
                language,
                normalized_surface,
                display_surface,
                encode(dict(morphology_features)),
                _clamp(morphology_confidence),
                now,
                now,
            ),
        )

    @staticmethod
    def _link_lexeme(
        conn: Any,
        *,
        lexeme_entity_id: str,
        language: str,
        canonical_lemma: str,
        sense_cluster_id: str,
        observed_at: Optional[str] = None,
    ) -> None:
        now = observed_at or utcnow()
        conn.execute(
            """INSERT INTO lexemes
               (lexeme_entity_id,language,canonical_lemma,sense_cluster_id,
                created_at,updated_at)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(lexeme_entity_id) DO UPDATE SET
                 updated_at=excluded.updated_at""",
            (
                lexeme_entity_id,
                language,
                canonical_lemma,
                sense_cluster_id,
                now,
                now,
            ),
        )

    @staticmethod
    def _link_word_usage(
        conn: Any,
        *,
        usage_occurrence_id: str,
        lexeme_entity_id: str,
        word_form_entity_id: str,
        source_id: str,
        sentence_index: int,
        token_index: int,
        observed_at: Optional[str] = None,
    ) -> None:
        conn.execute(
            """INSERT OR IGNORE INTO word_usages
               (usage_occurrence_id,lexeme_entity_id,word_form_entity_id,source_id,
                sentence_index,token_index,created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (
                usage_occurrence_id,
                lexeme_entity_id,
                word_form_entity_id,
                source_id,
                sentence_index,
                token_index,
                observed_at or utcnow(),
            ),
        )

    def _observe(
        self, conn: Any, universe_id: str, key: str, display: str, source_id: str,
        context_id: str, observable: Mapping[str, float], *, confidence: float,
        parent_occurrence_id: Optional[str] = None,
        observed_at: Optional[str] = None,
        event_rows: Optional[list[tuple[Any, ...]]] = None,
        entity_states: Optional[
            dict[str, tuple[int, Mapping[str, float]]]
        ] = None,
    ) -> tuple[str, str]:
        entity_id = stable_id("universe-entity", universe_id, key)
        occurrence_id = stable_id("universe-occurrence", universe_id, source_id, context_id, entity_id)
        now = observed_at or utcnow()
        observable_json = encode(dict(observable))
        cached_state = (
            entity_states.get(entity_id)
            if entity_states is not None
            else None
        )
        created_entity = None
        if cached_state is None:
            created_entity = conn.execute(
                """INSERT INTO universe_entities
                   (id,universe_id,observable_key,display_value,
                    prototype_vector_json,base_position_json,mass,gravity,
                    stability,frequency,dispersion,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,1,0,0,0,0,?,?)
                   ON CONFLICT(id) DO NOTHING
                   RETURNING id""",
                (
                    entity_id,
                    universe_id,
                    key,
                    display,
                    observable_json,
                    encode(_position(entity_id)),
                    now,
                    now,
                ),
            ).fetchone()
        if created_entity is not None:
            event_row = (
                stable_id(
                    "universe-history",
                    universe_id,
                    "entity_created",
                    entity_id,
                    now,
                ),
                universe_id,
                "entity_created",
                entity_id,
                encode({"source_id": source_id}),
                now,
            )
            if event_rows is None:
                conn.execute(
                    """INSERT INTO universe_training_events
                       (id,universe_id,event_type,subject_id,payload_json,
                        created_at)
                       VALUES(?,?,?,?,?,?)""",
                    event_row,
                )
            else:
                event_rows.append(event_row)
        inserted = conn.execute(
            """INSERT OR IGNORE INTO universe_occurrences
               (id,universe_id,entity_id,source_id,parent_occurrence_id,context_id,
                observable_features_json,context_vector_json,base_position_json,confidence,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                occurrence_id,
                universe_id,
                entity_id,
                source_id,
                parent_occurrence_id,
                context_id,
                observable_json,
                observable_json,
                encode(_position(occurrence_id)),
                _clamp(confidence),
                now,
            ),
        ).rowcount
        if inserted:
            if cached_state is not None:
                previous_frequency, previous_prototype = cached_state
            elif created_entity is not None:
                previous_frequency = 0
                previous_prototype = dict(observable)
            else:
                row = conn.execute(
                    """SELECT frequency,prototype_vector_json
                       FROM universe_entities WHERE id=?""",
                    (entity_id,),
                ).fetchone()
                previous_frequency = int(row["frequency"])
                previous_prototype = decode(
                    row["prototype_vector_json"], {}
                )
            frequency = previous_frequency + 1
            prototype = _mean_vectors(
                (previous_prototype, observable)
            )
            conn.execute(
                """UPDATE universe_entities SET prototype_vector_json=?,mass=?,gravity=?,stability=?,frequency=?,updated_at=? WHERE id=?""",
                (encode(prototype), round(1.0 + math.log1p(frequency), 6), round(min(1.0, frequency / 16.0), 6),
                 round(min(1.0, frequency / 8.0), 6), frequency, now, entity_id),
            )
            if entity_states is not None:
                entity_states[entity_id] = (frequency, prototype)
        return entity_id, occurrence_id

    @staticmethod
    def _observe_many(
        conn: Any,
        universe_id: Optional[str],
        source_id: str,
        observations: Sequence[Mapping[str, Any]],
        *,
        observed_at: str,
        event_rows: list[tuple[Any, ...]],
        entity_states: dict[str, tuple[int, Mapping[str, float]]],
    ) -> list[tuple[str, str]]:
        """Pre-aggregate a homogeneous observation batch.

        Symbols and fragments account for most occurrence writes.  Preparing
        their stable IDs first lets SQLite read existing aggregate state once
        and update each entity once while preserving the same observation
        order and recursive prototype calculation as ``_observe``.
        """
        prepared: list[dict[str, Any]] = []
        for observation in observations:
            item_universe_id = str(
                observation.get("universe_id") or universe_id or ""
            )
            if not item_universe_id:
                raise ValueError("observation universe_id is required")
            key = str(observation["key"])
            context_id = str(observation["context_id"])
            observable = {
                str(name): float(value)
                for name, value in dict(
                    observation.get("observable") or {}
                ).items()
            }
            entity_id = stable_id(
                "universe-entity", item_universe_id, key
            )
            occurrence_id = stable_id(
                "universe-occurrence",
                item_universe_id,
                source_id,
                context_id,
                entity_id,
            )
            prepared.append({
                "entity_id": entity_id,
                "occurrence_id": occurrence_id,
                "universe_id": item_universe_id,
                "key": key,
                "display": str(observation.get("display") or key),
                "context_id": context_id,
                "observable": observable,
                "observable_json": encode(observable),
                "confidence": _clamp(
                    float(observation.get("confidence") or 0.0)
                ),
                "parent_occurrence_id": observation.get(
                    "parent_occurrence_id"
                ),
            })
        if not prepared:
            return []

        entity_ids = sorted({
            str(item["entity_id"]) for item in prepared
            if str(item["entity_id"]) not in entity_states
        })
        if entity_ids:
            rows = conn.execute(
                """SELECT id,frequency,prototype_vector_json
                   FROM universe_entities WHERE id IN ({})""".format(
                    ",".join("?" for _ in entity_ids)
                ),
                entity_ids,
            ).fetchall()
            entity_states.update({
                str(row["id"]): (
                    int(row["frequency"]),
                    decode(row["prototype_vector_json"], {}),
                )
                for row in rows
            })

        first_by_entity: dict[str, Mapping[str, Any]] = {}
        for item in prepared:
            first_by_entity.setdefault(str(item["entity_id"]), item)
        missing = [
            item for entity_id, item in first_by_entity.items()
            if entity_id not in entity_states
        ]
        if missing:
            conn.executemany(
                """INSERT INTO universe_entities
                   (id,universe_id,observable_key,display_value,
                    prototype_vector_json,base_position_json,mass,gravity,
                    stability,frequency,dispersion,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,1,0,0,0,0,?,?)""",
                [
                    (
                        item["entity_id"],
                        item["universe_id"],
                        item["key"],
                        item["display"],
                        item["observable_json"],
                        encode(_position(str(item["entity_id"]))),
                        observed_at,
                        observed_at,
                    )
                    for item in missing
                ],
            )
            for item in missing:
                entity_id = str(item["entity_id"])
                entity_states[entity_id] = (
                    0,
                    dict(item["observable"]),
                )
                event_rows.append((
                    stable_id(
                        "universe-history",
                        item["universe_id"],
                        "entity_created",
                        entity_id,
                        observed_at,
                    ),
                    item["universe_id"],
                    "entity_created",
                    entity_id,
                    encode({"source_id": source_id}),
                    observed_at,
                ))

        occurrence_ids = [
            str(item["occurrence_id"]) for item in prepared
        ]
        existing_occurrences = {
            str(row["id"])
            for row in conn.execute(
                """SELECT id FROM universe_occurrences
                   WHERE id IN ({})""".format(
                    ",".join("?" for _ in occurrence_ids)
                ),
                occurrence_ids,
            ).fetchall()
        }
        inserted = [
            item for item in prepared
            if str(item["occurrence_id"]) not in existing_occurrences
        ]
        if inserted:
            conn.executemany(
                """INSERT INTO universe_occurrences
                   (id,universe_id,entity_id,source_id,
                    parent_occurrence_id,context_id,
                    observable_features_json,context_vector_json,
                    base_position_json,confidence,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        item["occurrence_id"],
                        item["universe_id"],
                        item["entity_id"],
                        source_id,
                        item["parent_occurrence_id"],
                        item["context_id"],
                        item["observable_json"],
                        item["observable_json"],
                        encode(_position(str(item["occurrence_id"]))),
                        item["confidence"],
                        observed_at,
                    )
                    for item in inserted
                ],
            )
            updated_ids: set[str] = set()
            for item in inserted:
                entity_id = str(item["entity_id"])
                frequency, prototype = entity_states[entity_id]
                entity_states[entity_id] = (
                    frequency + 1,
                    _mean_vectors((
                        prototype,
                        item["observable"],
                    )),
                )
                updated_ids.add(entity_id)
            conn.executemany(
                """UPDATE universe_entities SET prototype_vector_json=?,
                   mass=?,gravity=?,stability=?,frequency=?,updated_at=?
                   WHERE id=?""",
                [
                    (
                        encode(entity_states[entity_id][1]),
                        round(
                            1.0
                            + math.log1p(entity_states[entity_id][0]),
                            6,
                        ),
                        round(
                            min(
                                1.0,
                                entity_states[entity_id][0] / 16.0,
                            ),
                            6,
                        ),
                        round(
                            min(
                                1.0,
                                entity_states[entity_id][0] / 8.0,
                            ),
                            6,
                        ),
                        entity_states[entity_id][0],
                        observed_at,
                        entity_id,
                    )
                    for entity_id in sorted(updated_ids)
                ],
            )
        return [
            (str(item["entity_id"]), str(item["occurrence_id"]))
            for item in prepared
        ]

    @staticmethod
    def _write_transitions(
        conn: Any,
        transitions: Mapping[tuple[str, ...], int],
        *,
        observed_at: str,
        event_rows: list[tuple[Any, ...]],
    ) -> None:
        """Write one pre-aggregated transition batch for a source."""
        prepared = []
        for transition, evidence_count in transitions.items():
            (
                source_universe,
                target_universe,
                source_type,
                source_id,
                target_type,
                target_id,
                context_id,
            ) = transition
            transition_id = stable_id(
                "universe-transition",
                source_universe,
                target_universe,
                source_type,
                source_id,
                target_type,
                target_id,
                context_id,
            )
            count = max(1, int(evidence_count))
            prepared.append((
                transition_id,
                source_universe,
                target_universe,
                source_type,
                source_id,
                target_type,
                target_id,
                min(1.0, 0.7 + 0.05 * (count - 1)),
                min(1.0, 0.7 + 0.03 * (count - 1)),
                context_id,
                count,
                observed_at,
                observed_at,
            ))
        if not prepared:
            return
        prepared.sort(key=lambda row: row[0])
        existing_ids: set[str] = set()
        for start in range(0, len(prepared), 500):
            identifiers = [row[0] for row in prepared[start:start + 500]]
            existing_ids.update(
                str(row["id"])
                for row in conn.execute(
                    """SELECT id FROM universe_transitions
                       WHERE id IN ({})""".format(
                        ",".join("?" for _ in identifiers)
                    ),
                    identifiers,
                ).fetchall()
            )
        conn.executemany(
            """INSERT INTO universe_transitions
               (id,source_universe_id,target_universe_id,source_type,
                source_id,target_type,target_id,weight,confidence,context_id,
                evidence_count,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 evidence_count=evidence_count+excluded.evidence_count,
                 weight=MIN(1.0,weight+.05*excluded.evidence_count),
                 confidence=MIN(1.0,confidence+.03*excluded.evidence_count),
                 updated_at=excluded.updated_at""",
            prepared,
        )
        event_rows.extend(
            (
                stable_id(
                    "universe-history",
                    row[1],
                    "transition_created",
                    row[0],
                    observed_at,
                ),
                row[1],
                "transition_created",
                row[0],
                encode({"target_universe": row[2]}),
                observed_at,
            )
            for row in prepared
            if row[0] not in existing_ids
        )

    def _transition(
        self, conn: Any, source_universe: str, target_universe: str, source_type: str,
        source_id: str, target_type: str, target_id: str, context_id: str,
    ) -> None:
        transition_id = stable_id("universe-transition", source_universe, target_universe, source_type, source_id, target_type, target_id, context_id)
        now = utcnow()
        exists = conn.execute("SELECT evidence_count FROM universe_transitions WHERE id=?", (transition_id,)).fetchone()
        conn.execute(
            """INSERT INTO universe_transitions
               (id,source_universe_id,target_universe_id,source_type,source_id,target_type,target_id,
                weight,confidence,context_id,evidence_count,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,.7,.7,?,1,?,?)
               ON CONFLICT(id) DO UPDATE SET evidence_count=evidence_count+1,
                 weight=MIN(1.0,weight+.05),confidence=MIN(1.0,confidence+.03),updated_at=excluded.updated_at""",
            (transition_id, source_universe, target_universe, source_type, source_id, target_type, target_id, context_id, now, now),
        )
        if exists is None:
            _record_event(conn, source_universe, "transition_created", transition_id, {"target_universe": target_universe})

    def _refresh_clouds(self, conn: Any, universe_id: str) -> None:
        dimensions = conn.execute(
            "SELECT id,basis_json,status,stability FROM latent_dimensions WHERE universe_id=? AND status<>'pruned'", (universe_id,)
        ).fetchall()
        now = utcnow()
        for dimension in dimensions:
            members = conn.execute(
                "SELECT source_id,membership FROM projections WHERE dimension_id=? AND source_type='entity' AND membership>=.2", (dimension["id"],)
            ).fetchall()
            if not members:
                continue
            cloud_id = stable_id("entity-cloud", universe_id, dimension["id"])
            density = sum(float(row["membership"]) for row in members) / len(members)
            core_ids = [str(row["source_id"]) for row in sorted(members, key=lambda row: -float(row["membership"]))[:3]]
            existed = conn.execute("SELECT id FROM entity_clouds WHERE id=?", (cloud_id,)).fetchone()
            conn.execute(
                """INSERT INTO entity_clouds
                   (id,universe_id,core_vector_json,core_entity_ids_json,mass,gravity,radius,density,
                    dispersion,stability,member_count,status,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,0,?,?,?, ?,?)
                   ON CONFLICT(id) DO UPDATE SET core_entity_ids_json=excluded.core_entity_ids_json,
                     mass=excluded.mass,gravity=excluded.gravity,radius=excluded.radius,density=excluded.density,
                     stability=excluded.stability,member_count=excluded.member_count,status=excluded.status,updated_at=excluded.updated_at""",
                (cloud_id, universe_id, dimension["basis_json"], encode(core_ids), len(members), density,
                 1.0 - density, density, float(dimension["stability"]), len(members),
                 "active" if len(members) >= 2 else "candidate", now, now),
            )
            for member in members:
                membership = float(member["membership"])
                conn.execute(
                    """INSERT INTO cloud_memberships(cloud_id,source_type,source_id,membership,radial_distance,confidence)
                       VALUES(?, 'entity',?,?,?,.8)
                       ON CONFLICT(cloud_id,source_type,source_id) DO UPDATE SET membership=excluded.membership,
                         radial_distance=excluded.radial_distance,confidence=excluded.confidence""",
                    (cloud_id, member["source_id"], membership, 1.0 - membership),
                )
            if existed is None:
                _record_event(conn, universe_id, "cloud_created", cloud_id, {"dimension_id": dimension["id"]})

    def _refresh_statistics(self, conn: Any, universe_id: str) -> None:
        row = conn.execute(
            """SELECT
                 (SELECT COUNT(*) FROM universe_entities WHERE universe_id=?) AS entity_count,
                 (SELECT COUNT(*) FROM universe_occurrences WHERE universe_id=?) AS occurrence_count,
                 (SELECT COUNT(*) FROM entity_clouds WHERE universe_id=? AND status<>'pruned') AS cloud_count,
                 (SELECT COUNT(*) FROM latent_dimensions WHERE universe_id=? AND status<>'pruned') AS dimension_count""",
            (universe_id, universe_id, universe_id, universe_id),
        ).fetchone()
        conn.execute("UPDATE universes SET statistics_json=?,updated_at=? WHERE id=?", (encode(dict(row)), utcnow(), universe_id))

    def list_universes(self) -> dict[str, Any]:
        with self.repository.transaction() as conn:
            rows = conn.execute("SELECT * FROM universes ORDER BY rowid").fetchall()
            result = []
            for row in rows:
                stats = decode(row["statistics_json"], {})
                active = conn.execute("SELECT COUNT(*) FROM latent_dimensions WHERE universe_id=? AND status IN ('active','shared')", (row["id"],)).fetchone()[0]
                candidates = conn.execute("SELECT COUNT(*) FROM latent_dimensions WHERE universe_id=? AND status='candidate'", (row["id"],)).fetchone()[0]
                stability = conn.execute("SELECT COALESCE(AVG(stability),0) FROM universe_entities WHERE universe_id=?", (row["id"],)).fetchone()[0]
                result.append({
                    "id": row["id"],
                    "name": row["name"],
                    "scale": row["scale"],
                    "entity_count": int(stats.get("entity_count", 0)),
                    "occurrence_count": int(stats.get("occurrence_count", 0)),
                    "cloud_count": int(stats.get("cloud_count", 0)),
                    "dimension_count": int(stats.get("dimension_count", 0)),
                    "active_dimension_count": int(active),
                    "candidate_dimension_count": int(candidates),
                    "last_swarm_run_id": stats.get("last_swarm_run_id"),
                    "visited_in_last_query": bool(
                        stats.get("visited_in_last_query", False)
                    ),
                    "last_query_bee_count": int(
                        stats.get("last_query_bee_count", 0)
                    ),
                    "active_bee_count": int(
                        stats.get("active_bee_count", 0)
                    ),
                    "successful_bee_count": int(
                        stats.get("successful_bee_count", 0)
                    ),
                    "terminated_bee_count": int(
                        stats.get("terminated_bee_count", 0)
                    ),
                    "fallback_bee_count": int(
                        stats.get("fallback_bee_count", 0)
                    ),
                    "retrieval_mode": stats.get("retrieval_mode"),
                    "bee_count": int(stats.get("last_query_bee_count", 0)),
                    "stability": round(float(stability or 0), 6),
                })
            return {"universes": result}

    def base_space(self, universe_id: str, *, limit: int = 200, min_mass: float = 0.0, min_stability: float = 0.0, selected_context: str = "") -> dict[str, Any]:
        with self.repository.transaction() as conn:
            self._require_universe(conn, universe_id)
            entities = conn.execute(
                """SELECT * FROM universe_entities WHERE universe_id=? AND mass>=? AND stability>=?
                   ORDER BY mass DESC,observable_key LIMIT ?""", (universe_id, min_mass, min_stability, max(1, min(limit, 2000))),
            ).fetchall()
            entity_rows = [self._entity(row) for row in entities]
            clouds = [self._cloud(row) for row in conn.execute("SELECT * FROM entity_clouds WHERE universe_id=? AND status<>'pruned' ORDER BY density DESC", (universe_id,)).fetchall()]
            occurrences: list[dict[str, Any]] = []
            if selected_context:
                occurrences = [self._occurrence(row) for row in conn.execute(
                    "SELECT * FROM universe_occurrences WHERE universe_id=? AND context_id=? LIMIT ?", (universe_id, selected_context, max(1, min(limit, 2000))),
                ).fetchall()]
            return {"universe_id": universe_id, "space_type": "base", "entities": entity_rows,
                    "occurrences": occurrences, "entity_clouds": clouds,
                    "projected_positions": {row["id"]: row["base_position"][:2] for row in entity_rows},
                    "selected_context": selected_context or None}

    def dimensions(self, universe_id: str, *, status: str = "", scope: str = "", min_stability: float = 0.0, min_utility: float = 0.0, owner_cloud_id: str = "") -> dict[str, Any]:
        clauses, params = ["universe_id=?", "stability>=?"], [universe_id, min_stability]
        if status:
            clauses.append("status=?")
            params.append(status)
        if scope:
            clauses.append("owner_scope=?")
            params.append(scope)
        if owner_cloud_id:
            clauses.append("owner_id=?")
            params.append(owner_cloud_id)
        with self.repository.transaction() as conn:
            self._require_universe(conn, universe_id)
            rows = conn.execute("SELECT d.*,a.alias FROM latent_dimensions d LEFT JOIN dimension_aliases a ON a.dimension_id=d.id WHERE " + " AND ".join(clauses) + " ORDER BY stability DESC,evidence_count DESC", params).fetchall()
            values = [self._dimension(row) for row in rows]
            return {"universe_id": universe_id, "dimensions": [value for value in values if value["utility"] >= min_utility]}

    def dimension(self, dimension_id: str) -> dict[str, Any]:
        with self.repository.transaction() as conn:
            row = conn.execute("SELECT d.*,a.alias FROM latent_dimensions d LEFT JOIN dimension_aliases a ON a.dimension_id=d.id WHERE d.id=?", (dimension_id,)).fetchone()
            if row is None:
                raise KeyError(dimension_id)
            core = conn.execute("SELECT * FROM dimension_clouds WHERE dimension_id=?", (dimension_id,)).fetchone()
            projections = conn.execute(
                """SELECT p.*,e.display_value FROM projections p JOIN universe_entities e ON e.id=p.source_id
                   WHERE p.dimension_id=? AND p.source_type='entity' ORDER BY p.membership DESC LIMIT 12""", (dimension_id,),
            ).fetchall()
            relations = conn.execute("SELECT * FROM dimension_relations WHERE source_dimension_id=? OR target_dimension_id=?", (dimension_id, dimension_id)).fetchall()
            lineage = conn.execute(
                """SELECT * FROM dimension_lineage
                   WHERE canonical_dimension_id=?""",
                (row["canonical_dimension_id"],),
            ).fetchone()
            evaluations = conn.execute(
                """SELECT * FROM dimension_evaluations
                   WHERE dimension_id=? ORDER BY created_at DESC""",
                (dimension_id,),
            ).fetchall()
            shadows = conn.execute(
                """SELECT * FROM shadow_retrieval_runs
                   WHERE dimension_id=? ORDER BY created_at DESC LIMIT 20""",
                (dimension_id,),
            ).fetchall()
            history = conn.execute(
                """SELECT * FROM dimension_history
                   WHERE dimension_id=? ORDER BY created_at DESC""",
                (dimension_id,),
            ).fetchall()
            return {
                "metadata": self._dimension(row),
                "representation_type": row["representation_type"],
                "scope": {
                    "owner_scope": row["owner_scope"],
                    "owner_id": row["owner_id"],
                },
                "semantic_basis": decode(row["basis_json"], {}),
                "control_features": decode(
                    row["basis_json"], {}
                ).get("control_features_used", []),
                "core": decode(core["core_vector_json"], {}) if core else {},
                "core_examples": [
                    self._projection(item) for item in projections
                    if float(item["membership"]) >= .66
                ],
                "peripheral_examples": [
                    self._projection(item) for item in projections
                    if .3 <= float(item["membership"]) < .66
                ],
                "boundary_examples": [
                    self._projection(item) for item in projections
                    if float(item["membership"]) < .3
                ],
                "related_dimensions": [dict(item) for item in relations],
                "lineage": (
                    self._export_row(lineage) if lineage else None
                ),
                "evaluations": [
                    self._export_row(item) for item in evaluations
                ],
                "shadow_evaluations": [
                    self._export_row(item) for item in shadows
                ],
                "history": [
                    self._export_row(item) for item in history
                ],
            }

    def projections(self, dimension_id: str, *, source_type: str = "", limit: int = 100, min_membership: float = 0.0, context_id: str = "", sort: str = "membership") -> dict[str, Any]:
        if source_type and source_type not in {"entity", "occurrence"}:
            raise ValueError("source_type must be entity or occurrence")
        order = "membership DESC" if sort != "distance" else "distance_to_core ASC"
        with self.repository.transaction() as conn:
            exists = conn.execute("SELECT 1 FROM latent_dimensions WHERE id=?", (dimension_id,)).fetchone()
            if not exists:
                raise KeyError(dimension_id)
            clauses, params = ["dimension_id=?", "membership>=?"], [dimension_id, min_membership]
            if source_type:
                clauses.append("source_type=?")
                params.append(source_type)
            if context_id:
                clauses.append("context_id=?")
                params.append(context_id)
            rows = conn.execute("SELECT * FROM projections WHERE " + " AND ".join(clauses) + " ORDER BY " + order + " LIMIT ?", [*params, max(1, min(limit, 2000))]).fetchall()
            return {"dimension_id": dimension_id, "projections": [self._projection(row) for row in rows]}

    def profile(self, entity_id: str) -> dict[str, Any]:
        with self.repository.transaction() as conn:
            entity = conn.execute("SELECT * FROM universe_entities WHERE id=?", (entity_id,)).fetchone()
            if entity is None:
                raise KeyError(entity_id)
            clouds = conn.execute("SELECT c.*,m.membership,m.radial_distance FROM cloud_memberships m JOIN entity_clouds c ON c.id=m.cloud_id WHERE m.source_type='entity' AND m.source_id=? ORDER BY m.membership DESC", (entity_id,)).fetchall()
            stable = conn.execute("SELECT p.*,d.status,d.stability,d.strength FROM projections p JOIN latent_dimensions d ON d.id=p.dimension_id WHERE p.source_type='entity' AND p.source_id=? ORDER BY p.membership DESC", (entity_id,)).fetchall()
            occurrences = conn.execute("SELECT * FROM universe_occurrences WHERE entity_id=? ORDER BY created_at DESC", (entity_id,)).fetchall()
            contextual = conn.execute("SELECT p.* FROM projections p WHERE p.source_type='occurrence' AND p.source_id IN (SELECT id FROM universe_occurrences WHERE entity_id=?) ORDER BY p.membership DESC", (entity_id,)).fetchall()
            result = {
                "entity": self._entity(entity),
                "base_position": decode(entity["base_position_json"], []),
                "entity_clouds": [{**self._cloud(row), "membership": row["membership"], "radial_distance": row["radial_distance"]} for row in clouds],
                "stable_dimensions": [self._projection(row) for row in stable],
                "contextual_dimensions": [self._projection(row) for row in contextual],
                "occurrence_distribution": [self._occurrence(row) for row in occurrences],
            }
            if str(entity["universe_id"]) == "words":
                lexeme = conn.execute(
                    """SELECT canonical_lemma,sense_cluster_id FROM lexemes
                       WHERE lexeme_entity_id=?""",
                    (entity_id,),
                ).fetchone()
                forms = conn.execute(
                    """SELECT f.*,e.id,e.universe_id,e.observable_key,e.display_value,
                              e.frequency,e.mass
                       FROM word_forms f
                       JOIN universe_entities e ON e.id=f.word_form_entity_id
                       WHERE f.lexeme_entity_id=?
                       ORDER BY e.frequency DESC,f.normalized_surface""",
                    (entity_id,),
                ).fetchall()
                result.update({
                    "canonical_lemma": str(
                        lexeme["canonical_lemma"]
                        if lexeme is not None else entity["display_value"]
                    ),
                    "sense_cluster_id": (
                        (str(lexeme["sense_cluster_id"]) or None)
                        if lexeme is not None
                        else None
                    ),
                    "word_forms": [
                        {
                            "id": row["word_form_entity_id"],
                            "surface": row["display_surface"],
                            "normalized_surface": row["normalized_surface"],
                            "morphological_features": decode(row["morphological_features_json"], {}),
                            "morphology_confidence": float(row["morphology_confidence"]),
                            "usage_count": int(row["frequency"]),
                        }
                        for row in forms
                    ],
                    "usage_count": int(conn.execute(
                        "SELECT COUNT(*) FROM word_usages WHERE lexeme_entity_id=?",
                        (entity_id,),
                    ).fetchone()[0]),
                })
            elif str(entity["universe_id"]) == "word_forms":
                form = conn.execute(
                    """SELECT f.*,e.display_value AS lexeme
                       FROM word_forms f JOIN universe_entities e
                         ON e.id=f.lexeme_entity_id
                       WHERE f.word_form_entity_id=?""",
                    (entity_id,),
                ).fetchone()
                if form is not None:
                    result.update({
                        "lexeme_id": form["lexeme_entity_id"],
                        "canonical_lemma": form["lexeme"],
                        "morphological_features": decode(
                            form["morphological_features_json"], {}
                        ),
                        "morphology_confidence": float(
                            form["morphology_confidence"]
                        ),
                    })
            return result

    def compare(self, entity_ids: Sequence[str], universe_id: str) -> dict[str, Any]:
        if len(entity_ids) != 2:
            raise ValueError("exactly two entity_ids are required")
        with self.repository.transaction() as conn:
            rows = [conn.execute("SELECT * FROM universe_entities WHERE id=? AND universe_id=?", (entity_id, universe_id)).fetchone() for entity_id in entity_ids]
            if any(row is None for row in rows):
                raise KeyError("entity not found in universe")
            left, right = rows
            left_vector, right_vector = decode(left["prototype_vector_json"], {}), decode(right["prototype_vector_json"], {})
            left_projection = {row["dimension_id"]: float(row["membership"]) for row in conn.execute("SELECT dimension_id,membership FROM projections WHERE source_type='entity' AND source_id=?", (left["id"],)).fetchall()}
            right_projection = {row["dimension_id"]: float(row["membership"]) for row in conn.execute("SELECT dimension_id,membership FROM projections WHERE source_type='entity' AND source_id=?", (right["id"],)).fetchall()}
            shared = sorted(set(left_projection) & set(right_projection), key=lambda key: -min(left_projection[key], right_projection[key]))
            comparison = [{"dimension_id": key, "left": left_projection[key], "right": right_projection[key], "difference": abs(left_projection[key] - right_projection[key])} for key in shared]
            return {"universe_id": universe_id, "entities": [self._entity(left), self._entity(right)],
                    "base_distance": round(1.0 - _similarity(left_vector, right_vector), 6),
                    "shared_dimensions": [item for item in comparison if item["difference"] <= .25],
                    "different_dimensions": [item for item in comparison if item["difference"] > .25],
                    "projection_comparison": comparison, "shared_clouds": self._shared_clouds(conn, str(left["id"]), str(right["id"])),
                    "contextual_differences": []}

    def project(self, universe_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        space_type = str(payload.get("space_type") or "base")
        if space_type == "base":
            return self.base_space(universe_id, limit=int(payload.get("limit") or 200))
        dimensions = [str(item) for item in payload.get("dimension_ids", [])]
        if not dimensions:
            raise ValueError("dimension_ids are required outside base space")
        result = []
        with self.repository.transaction() as conn:
            self._require_universe(conn, universe_id)
            entities = conn.execute("SELECT id,display_value FROM universe_entities WHERE universe_id=? ORDER BY frequency DESC LIMIT ?", (universe_id, max(1, min(int(payload.get("limit") or 200), 2000)))).fetchall()
            entity_ids = [str(entity["id"]) for entity in entities]
            memberships_by_entity: dict[str, dict[str, float]] = defaultdict(dict)
            if entity_ids:
                rows = conn.execute(
                    "SELECT source_id,dimension_id,membership FROM projections WHERE source_type='entity' AND source_id IN ({}) AND dimension_id IN ({})".format(
                        ",".join("?" for _ in entity_ids), ",".join("?" for _ in dimensions)
                    ),
                    [*entity_ids, *dimensions],
                ).fetchall()
                for row in rows:
                    memberships_by_entity[str(row["source_id"])][str(row["dimension_id"])] = float(row["membership"])
            for entity in entities:
                memberships = memberships_by_entity[str(entity["id"])]
                result.append({"id": entity["id"], "label": entity["display_value"], "x": memberships.get(dimensions[0], 0.0), "y": memberships.get(dimensions[1], 0.0) if len(dimensions) > 1 else 0.5, "projections": memberships})
        return {"universe_id": universe_id, "space_type": space_type, "projection_method": payload.get("projection_method") or "selected_dimensions", "points": result, "notice": "Screen coordinates are a two-dimensional projection, not the model distance."}

    def transitions(self, universe_id: str, entity_id: str = "", limit: int = 200) -> dict[str, Any]:
        with self.repository.transaction() as conn:
            clauses, params = ["source_universe_id=?"], [universe_id]
            if entity_id:
                clauses.append("source_id=?")
                params.append(entity_id)
            rows = conn.execute("SELECT * FROM universe_transitions WHERE " + " AND ".join(clauses) + " ORDER BY weight DESC LIMIT ?", [*params, max(1, min(limit, 2000))]).fetchall()
            return {"transitions": [dict(row) for row in rows]}

    def history(self, universe_id: str = "", dimension_id: str = "", limit: int = 200) -> dict[str, Any]:
        with self.repository.transaction() as conn:
            clauses, params = [], []
            if universe_id:
                clauses.append("universe_id=?")
                params.append(universe_id)
            if dimension_id:
                clauses.append("subject_id=?")
                params.append(dimension_id)
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            rows = conn.execute("SELECT * FROM universe_training_events" + where + " ORDER BY created_at DESC LIMIT ?", [*params, max(1, min(limit, 2000))]).fetchall()
            return {"events": [{**dict(row), "payload": decode(row["payload_json"], {})} for row in rows]}

    def set_alias(self, dimension_id: str, alias: str) -> dict[str, Any]:
        alias = alias.strip()
        if not alias:
            raise ValueError("alias must not be empty")
        with self.repository.transaction() as conn:
            if not conn.execute("SELECT 1 FROM latent_dimensions WHERE id=?", (dimension_id,)).fetchone():
                raise KeyError(dimension_id)
            conn.execute("INSERT INTO dimension_aliases(dimension_id,alias,updated_at) VALUES(?,?,?) ON CONFLICT(dimension_id) DO UPDATE SET alias=excluded.alias,updated_at=excluded.updated_at", (dimension_id, alias, utcnow()))
        return {"dimension_id": dimension_id, "alias": alias}

    @staticmethod
    def _require_universe(conn: Any, universe_id: str) -> None:
        if not conn.execute("SELECT 1 FROM universes WHERE id=?", (universe_id,)).fetchone():
            raise KeyError(universe_id)

    @staticmethod
    def _entity(row: Mapping[str, Any]) -> dict[str, Any]:
        return {"id": row["id"], "universe_id": row["universe_id"], "key": row["observable_key"], "label": row["display_value"], "prototype_vector": decode(row["prototype_vector_json"], {}), "base_position": decode(row["base_position_json"], []), "mass": row["mass"], "gravity": row["gravity"], "stability": row["stability"], "frequency": row["frequency"], "dispersion": row["dispersion"]}

    @staticmethod
    def _occurrence(row: Mapping[str, Any]) -> dict[str, Any]:
        return {"id": row["id"], "universe_id": row["universe_id"], "entity_id": row["entity_id"], "source_id": row["source_id"], "parent_occurrence_id": row["parent_occurrence_id"], "context_id": row["context_id"], "observable_features": decode(row["observable_features_json"], {}), "context_vector": decode(row["context_vector_json"], {}), "base_position": decode(row["base_position_json"], []), "confidence": row["confidence"]}

    @staticmethod
    def _cloud(row: Mapping[str, Any]) -> dict[str, Any]:
        return {"id": row["id"], "universe_id": row["universe_id"], "core_vector": decode(row["core_vector_json"], {}), "core_entity_ids": decode(row["core_entity_ids_json"], []), "mass": row["mass"], "gravity": row["gravity"], "radius": row["radius"], "density": row["density"], "stability": row["stability"], "member_count": row["member_count"], "status": row["status"]}

    @staticmethod
    def _dimension(row: Mapping[str, Any]) -> dict[str, Any]:
        utility = _clamp(float(row["predictive_gain"]) + float(row["retrieval_gain"]) + float(row["compression_gain"]) - float(row["memory_cost"]))
        return {
            "id": row["id"],
            "canonical_dimension_id": row["canonical_dimension_id"],
            "revision": row["revision"],
            "universe_id": row["universe_id"],
            "alias": row["alias"] if "alias" in row.keys() else None,
            "owner_scope": row["owner_scope"],
            "owner_id": row["owner_id"],
            "representation_type": row["representation_type"],
            "basis": decode(row["basis_json"], {}),
            "dimensionality": row["dimensionality"],
            "strength": row["strength"],
            "stability": row["stability"],
            "stability_lower_bound": row["stability_lower_bound"],
            "predictive_gain": row["predictive_gain"],
            "retrieval_gain": row["retrieval_gain"],
            "holdout_retrieval_gain": row["holdout_retrieval_gain"],
            "shadow_retrieval_gain": row["shadow_retrieval_gain"],
            "compression_gain": row["compression_gain"],
            "memory_cost": row["memory_cost"],
            "usage_count": row["usage_count"],
            "projection_usage_count": row["projection_usage_count"],
            "retrieval_contribution_count": (
                row["retrieval_contribution_count"]
            ),
            "graph_admitted_contribution_count": (
                row["graph_admitted_contribution_count"]
            ),
            "validated_answer_contribution_count": (
                row["validated_answer_contribution_count"]
            ),
            "evidence_count": row["evidence_count"],
            "entity_support": row["entity_support"],
            "source_support": row["source_support"],
            "domain_support": row["domain_support"],
            "train_support": row["train_support"],
            "holdout_support": row["holdout_support"],
            "continual_support": row["continual_support"],
            "utility": utility,
            "status": row["status"],
            "created_at": row["created_at"],
            "activated_at": row["activated_at"],
            "last_updated_at": row["last_updated_at"],
        }

    @staticmethod
    def _projection(row: Mapping[str, Any]) -> dict[str, Any]:
        value = {"id": row["id"], "dimension_id": row["dimension_id"], "source_type": row["source_type"], "source_id": row["source_id"], "context_id": row["context_id"], "coordinates": decode(row["coordinates_json"], []), "membership": row["membership"], "distance_to_core": row["distance_to_core"], "confidence": row["confidence"]}
        if "display_value" in row.keys():
            value["label"] = row["display_value"]
        return value

    @staticmethod
    def _shared_clouds(conn: Any, left_id: str, right_id: str) -> list[dict[str, Any]]:
        rows = conn.execute(
            """SELECT a.cloud_id,a.membership AS left_membership,b.membership AS right_membership
               FROM cloud_memberships a JOIN cloud_memberships b ON b.cloud_id=a.cloud_id
               WHERE a.source_type='entity' AND b.source_type='entity' AND a.source_id=? AND b.source_id=?""", (left_id, right_id),
        ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _export_row(row: Mapping[str, Any]) -> dict[str, Any]:
        item = dict(row)
        for key, value in tuple(item.items()):
            if key.endswith("_json"):
                item[key[:-5]] = decode(value, {})
                del item[key]
        return item
