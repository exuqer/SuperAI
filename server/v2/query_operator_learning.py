"""Learn query-operator usages without assigning semantic labels.

The graph pipeline still owns the universal GAP mechanics.  This module only
keeps evidence about *uses* of an operator: which structural slots and answer
signatures were confirmed for an occurrence in a particular query context.
It intentionally never maps a surface such as ``кто`` to a named role or an
entity type.

The first integration is shadow-only.  Predictions are exposed in a
``QueryGraph`` trace and persisted for evaluation, but they are not a scoring
input for ``GraphMatcher`` yet.  That makes it possible to compare learned
profiles with the established morphology-based route before promoting them.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence

from .graph_models import AnswerStatus, CandidateBinding, GapNode, QueryGraph
from .graph_repository import decode, encode, stable_id, utcnow


PROFILE_PROJECTION_NAMES = (
    "lexical",
    "morphology",
    "query_context",
    "local_slot",
    "answer_cloud",
    "event_relation",
    "dialogue_context",
    "lineage",
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalise_surface(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _stable_dimension_id(namespace: str, values: Mapping[str, float]) -> str:
    """Return an opaque dimension label for trace inspection.

    The label is deliberately derived from observations rather than any human
    semantic category.  It is an inspection handle, not an ontology entry.
    """
    payload = json.dumps(
        sorted((str(key), round(float(value), 5)) for key, value in values.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"D-Q-{namespace}-{digest}"


def _average_distribution(
    previous: Mapping[str, float],
    observed: Mapping[str, float],
    previous_count: int,
) -> Dict[str, float]:
    """Update a sparse empirical distribution with one observation."""
    denominator = max(1, int(previous_count) + 1)
    return {
        key: _clamp(
            (
                float(previous.get(key, 0.0)) * max(0, int(previous_count))
                + float(observed.get(key, 0.0))
            ) / denominator
        )
        for key in set(previous) | set(observed)
    }


class QueryOperatorLearner:
    """Persistence and shadow prediction for concrete query-operator uses."""

    @staticmethod
    def _profile_key(gap: GapNode) -> str:
        # A profile starts at the surface level.  Its transferable content is
        # the contextual projection accumulated beneath that key; it is never
        # interpreted as a fixed semantic meaning for the word.
        return f"surface:{_normalise_surface(gap.surface)}"

    @staticmethod
    def _context_projections(
        graph: QueryGraph,
        gap: GapNode,
    ) -> Dict[str, Dict[str, float]]:
        surface = _normalise_surface(gap.surface)
        signature = gap.question_signature.as_dict()
        morphology = {
            key: value
            for key, value in signature.items()
            if key.startswith("morph:")
        }
        question_context = {
            f"gap_kind:{gap.gap_kind.value}": 1.0,
            f"known_node_count:{min(len(graph.known_nodes), 8)}": 1.0,
            f"target_gap_count:{min(len(tuple(graph.target_gaps) or (gap,)), 8)}": 1.0,
            f"operator_position:{min(gap.token_indices or (0,))}": 1.0,
        }
        continuation_mode = str(graph.trace.get("continuation_mode") or "NONE")
        return {
            "lexical": {f"surface:{surface}": 1.0} if surface else {},
            "morphology": morphology,
            "query_context": question_context,
            "local_slot": {},
            "answer_cloud": {},
            "event_relation": {f"gap_kind:{gap.gap_kind.value}": 1.0},
            "dialogue_context": {
                f"continuation:{continuation_mode}": 1.0,
            },
            "lineage": {
                "has_parent_query": 1.0 if graph.continuation_of else 0.0,
            },
        }

    @staticmethod
    def _answer_cloud(binding: CandidateBinding) -> Dict[str, float]:
        features = dict(binding.resolved_features)
        result: Dict[str, float] = {
            f"shape:surface_tokens:{min(len(binding.resolved_surface.split()), 4)}": 1.0,
        }
        for feature in ("case", "number", "gender", "animacy", "preposition"):
            value = str(features.get(feature) or "")
            if value:
                result[f"observable:{feature}:{value}"] = 1.0
        # The opaque signature captures a recurrent answer cloud without
        # baking a label such as "person" or "place" into the model.
        signature_payload = json.dumps(sorted(result), ensure_ascii=False)
        result[
            "answer_signature:"
            + hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()[:12]
        ] = 1.0
        return result

    @staticmethod
    def _binding_slot_distribution(binding: CandidateBinding) -> Dict[str, float]:
        slot_ids = sorted({
            str(slot_id)
            for evidence in binding.evidence
            for slot_id in evidence.get("local_slot_ids", [])
            if slot_id
        })
        if not slot_ids:
            return {}
        weight = 1.0 / len(slot_ids)
        return {slot_id: weight for slot_id in slot_ids}

    @staticmethod
    def _load_profile(conn: Any, profile_key: str) -> Optional[Mapping[str, Any]]:
        return conn.execute(
            """SELECT * FROM query_operator_profiles
               WHERE profile_key=?""",
            (profile_key,),
        ).fetchone()

    def predict(
        self,
        conn: Any,
        graph: QueryGraph,
        gap: GapNode,
    ) -> Dict[str, Any]:
        """Produce a non-operative prediction for one operator occurrence."""
        profile_key = self._profile_key(gap)
        context = self._context_projections(graph, gap)
        row = self._load_profile(conn, profile_key)
        if row is None:
            return {
                "mode": "SHADOW",
                "profile_key": profile_key,
                "profile_status": "UNSEEN",
                "support_count": 0,
                "confidence": 0.0,
                "compatible_local_slots": {},
                "projection_dimensions": {
                    namespace: {
                        "dimension_id": _stable_dimension_id(namespace, values),
                        "membership": 0.0,
                    }
                    for namespace, values in context.items()
                    if values
                },
            }
        projections = decode(row["projections_json"], {})
        local_slots = decode(row["compatible_slots_json"], {})
        context_fit = []
        for namespace in ("morphology", "query_context", "event_relation", "dialogue_context"):
            current = context.get(namespace, {})
            learned = projections.get(namespace, {})
            if not current:
                continue
            matching = sum(
                min(float(weight), float(learned.get(key, 0.0)))
                for key, weight in current.items()
            )
            context_fit.append(matching / max(1, len(current)))
        applicability = sum(context_fit) / len(context_fit) if context_fit else 0.0
        return {
            "mode": "SHADOW",
            "profile_id": str(row["id"]),
            "profile_key": profile_key,
            "profile_status": str(row["status"]),
            "support_count": int(row["support_count"]),
            "confidence": _clamp(float(row["confidence"])),
            "context_applicability": _clamp(applicability),
            "compatible_local_slots": {
                str(key): _clamp(value)
                for key, value in sorted(
                    local_slots.items(), key=lambda item: item[1], reverse=True
                )[:8]
            },
            "projection_dimensions": {
                namespace: {
                    "dimension_id": _stable_dimension_id(
                        namespace,
                        dict(projections.get(namespace, {})) or values,
                    ),
                    "membership": _clamp(
                        sum(float(value) for value in projections.get(namespace, {}).values())
                        / max(1, len(projections.get(namespace, {})))
                    ),
                }
                for namespace, values in context.items()
                if values or projections.get(namespace)
            },
        }

    def _upsert_profile(
        self,
        conn: Any,
        graph: QueryGraph,
        gap: GapNode,
        binding: CandidateBinding,
        *,
        accepted: bool,
    ) -> str:
        profile_key = self._profile_key(gap)
        row = self._load_profile(conn, profile_key)
        profile_id = str(row["id"]) if row else stable_id(
            "query-operator-profile", profile_key
        )
        previous_positive = int(row["validated_count"]) if row else 0
        previous_rejected = int(row["rejected_count"]) if row else 0
        previous_projections = decode(row["projections_json"], {}) if row else {}
        previous_slots = decode(row["compatible_slots_json"], {}) if row else {}
        observed = self._context_projections(graph, gap)
        observed["local_slot"] = self._binding_slot_distribution(binding)
        observed["answer_cloud"] = self._answer_cloud(binding)
        if accepted:
            projections = {
                namespace: _average_distribution(
                    dict(previous_projections.get(namespace, {})),
                    values,
                    previous_positive,
                )
                for namespace, values in observed.items()
            }
            slots = _average_distribution(
                previous_slots,
                observed["local_slot"],
                previous_positive,
            )
            validated_count = previous_positive + 1
            rejected_count = previous_rejected
        else:
            projections = previous_projections
            slots = previous_slots
            validated_count = previous_positive
            rejected_count = previous_rejected + 1
        support_count = validated_count + rejected_count
        confidence = validated_count / (support_count + 2.0)
        # This integration is intentionally shadow-only.  Repeated examples
        # increase support and confidence, but promotion needs the separate
        # transfer/holdout/continuation checks described by the model
        # invariant; raw support alone is not an activation criterion.
        status = "SHADOW"
        now = utcnow()
        conn.execute(
            """INSERT INTO query_operator_profiles
               (id,profile_key,projections_json,compatible_slots_json,
                support_count,validated_count,rejected_count,confidence,status,
                created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(profile_key) DO UPDATE SET
                 projections_json=excluded.projections_json,
                 compatible_slots_json=excluded.compatible_slots_json,
                 support_count=excluded.support_count,
                 validated_count=excluded.validated_count,
                 rejected_count=excluded.rejected_count,
                 confidence=excluded.confidence,status=excluded.status,
                 updated_at=excluded.updated_at""",
            (
                profile_id,
                profile_key,
                encode(projections),
                encode(slots),
                support_count,
                validated_count,
                rejected_count,
                confidence,
                status,
                str(row["created_at"]) if row else now,
                now,
            ),
        )
        return profile_id

    def record_outcomes(
        self,
        conn: Any,
        graph: QueryGraph,
        *,
        selected_bindings: Sequence[CandidateBinding],
        accepted_bindings: Sequence[CandidateBinding],
        rejected: Sequence[Mapping[str, Any]],
        answer: Mapping[str, Any],
        observational_only: bool = False,
    ) -> None:
        """Persist every occurrence and update profiles only after validation."""
        status = str(answer.get("status") or "")
        validation = dict(answer.get("validation") or {})
        confirmed = bool(validation.get("valid")) and status in {
            AnswerStatus.RESOLVED.value,
            AnswerStatus.PARTIALLY_RESOLVED.value,
        }
        selected_by_gap = {
            binding.gap_node_id: binding for binding in selected_bindings
        }
        accepted_by_gap: Dict[str, list[CandidateBinding]] = {}
        for binding in accepted_bindings:
            accepted_by_gap.setdefault(binding.gap_node_id, []).append(binding)

        for gap in tuple(graph.target_gaps):
            if not _normalise_surface(gap.surface):
                continue
            selected = selected_by_gap.get(gap.id)
            occurrence_id = stable_id("query-operator-occurrence", graph.id, gap.id)
            prediction = dict(gap.evidence.get("learned_gap_profile") or {})
            profile_id = (
                self._upsert_profile(conn, graph, gap, selected, accepted=confirmed)
                if selected and not observational_only else prediction.get("profile_id")
            )
            occurrence_status = (
                "OBSERVED_UNTRUSTED" if observational_only else
                "VALIDATED" if selected and confirmed else
                "REJECTED" if selected else
                "OBSERVED"
            )
            conn.execute(
                """INSERT OR REPLACE INTO query_operator_occurrences
                   (id,query_graph_id,gap_node_id,profile_id,operator_surface,
                    operator_normalized,token_indices_json,context_json,
                    prediction_json,status,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    occurrence_id,
                    graph.id,
                    gap.id,
                    profile_id,
                    gap.surface,
                    _normalise_surface(gap.surface),
                    encode(list(gap.token_indices)),
                    encode(self._context_projections(graph, gap)),
                    encode(prediction),
                    occurrence_status,
                    utcnow(),
                ),
            )
            conn.execute(
                "DELETE FROM query_operator_experiences WHERE occurrence_id=?",
                (occurrence_id,),
            )
            if selected:
                conn.execute(
                    """INSERT INTO query_operator_experiences
                       (id,occurrence_id,profile_id,outcome,validated,
                        binding_json,rejection_json,created_at)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        stable_id("query-operator-experience", occurrence_id, "selected"),
                        occurrence_id,
                        profile_id,
                        "OBSERVED_UNTRUSTED" if observational_only else
                        "VALIDATED_BINDING" if confirmed else "REJECTED_BINDING",
                        0 if observational_only else int(confirmed),
                        encode(selected.as_dict()),
                        "{}",
                        utcnow(),
                    ),
                )
            selected_id = selected.id if selected else ""
            for binding in accepted_by_gap.get(gap.id, []):
                if binding.id == selected_id:
                    continue
                conn.execute(
                    """INSERT INTO query_operator_experiences
                       (id,occurrence_id,profile_id,outcome,validated,
                        binding_json,rejection_json,created_at)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        stable_id("query-operator-experience", occurrence_id, binding.id),
                        occurrence_id,
                        profile_id,
                        "UNSELECTED_CANDIDATE",
                        0,
                        encode(binding.as_dict()),
                        "{}",
                        utcnow(),
                    ),
                )
            for index, rejection in enumerate(rejected):
                conn.execute(
                    """INSERT INTO query_operator_experiences
                       (id,occurrence_id,profile_id,outcome,validated,
                        binding_json,rejection_json,created_at)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        stable_id("query-operator-experience", occurrence_id, "rejected", index),
                        occurrence_id,
                        profile_id,
                        "REJECTED_EVENT",
                        0,
                        "{}",
                        encode(dict(rejection)),
                        utcnow(),
                    ),
                )
