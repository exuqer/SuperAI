"""Evidence-aware candidate relation analysis and grouping."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from .contracts import (
    BoundedAssociativeWorkspace,
    Candidate,
    CandidateCompatibility,
    CandidateRelation,
    Conflict,
    _id,
)


def _norm(value: Any) -> str:
    return str(value or "").casefold().replace("ё", "е").strip()


def _origin(candidate: Candidate, key: str) -> str:
    return _norm((candidate.origin or {}).get(key))


def _semantic_key(candidate: Candidate) -> tuple[str, str]:
    signature = candidate.structural_signature or {}
    lemma = _norm(candidate.lemma or candidate.surface or candidate.element_id)
    kind = _norm(
        signature.get("source_kind")
        or (candidate.origin or {}).get("semantic_kind")
        or (candidate.origin or {}).get("type")
        or "unknown"
    )
    return kind, lemma


def _signature_compatible(left: Candidate, right: Candidate) -> bool:
    left_sig = dict(left.structural_signature or {})
    right_sig = dict(right.structural_signature or {})
    if not left_sig or not right_sig:
        return True
    for key in (
        "source_kind",
        "attachment_form",
        "preposition",
        "position_relative_to_predicate",
        "relation_to_known_member",
        "surface_projection_kind",
    ):
        left_value = _norm(left_sig.get(key))
        right_value = _norm(right_sig.get(key))
        if left_value and right_value and left_value != right_value:
            return False
    return True


def _polarity(candidate: Candidate) -> str:
    value = _norm(candidate.polarity or "positive")
    if value in {"negative", "negated", "false", "not", "отрицательная"}:
        return "NEGATIVE"
    return "POSITIVE"


def _state(candidate: Candidate) -> str:
    origin = candidate.origin or {}
    for key in ("current_state", "state", "state_value", "actuality"):
        if origin.get(key):
            return _norm(origin[key])
    for item in candidate.provenance:
        if isinstance(item, Mapping):
            for key in ("current_state", "state", "state_value", "actuality"):
                if item.get(key):
                    return _norm(item[key])
    return ""


def _same_mention(left: Candidate, right: Candidate) -> bool:
    for key in ("mention_id", "attachment_id", "alternative_group", "interpretation_id"):
        left_value = _origin(left, key)
        right_value = _origin(right, key)
        if left_value and right_value and left_value == right_value:
            return True
    return False


def classify_candidate_relation(
    left: Candidate,
    right: Candidate,
    workspace: BoundedAssociativeWorkspace | None = None,
) -> CandidateRelation:
    if left.candidate_id == right.candidate_id:
        return CandidateRelation.DUPLICATE
    if left.gap_id != right.gap_id:
        return CandidateRelation.UNRELATED
    if left.element_id == right.element_id:
        return CandidateRelation.DUPLICATE
    if _polarity(left) != _polarity(right):
        return CandidateRelation.CONTRADICTORY
    if left.conflict_score >= 0.6 or right.conflict_score >= 0.6:
        return CandidateRelation.CONTRADICTORY
    left_conflicts = set(left.origin.get("conflicts") or ()) | set(left.provenance[0].get("conflicts") or () if left.provenance and isinstance(left.provenance[0], Mapping) else ())
    right_conflicts = set(right.origin.get("conflicts") or ()) | set(right.provenance[0].get("conflicts") or () if right.provenance and isinstance(right.provenance[0], Mapping) else ())
    if left.element_id in right_conflicts or right.element_id in left_conflicts or left_conflicts & right_conflicts:
        return CandidateRelation.CONTRADICTORY
    if _state(left) and _state(right) and _state(left) != _state(right):
        return CandidateRelation.CONTRADICTORY
    if _same_mention(left, right):
        return CandidateRelation.ALTERNATIVE
    same_sense = bool(_origin(left, "sense_id") and _origin(left, "sense_id") == _origin(right, "sense_id"))
    same_interpretation = bool(_origin(left, "interpretation_id") and _origin(left, "interpretation_id") == _origin(right, "interpretation_id"))
    if same_sense or same_interpretation:
        return CandidateRelation.ALTERNATIVE
    if not (left.evidence_ids or left.spatial_support_ids) or not (right.evidence_ids or right.spatial_support_ids):
        return CandidateRelation.UNKNOWN
    if not _signature_compatible(left, right):
        return CandidateRelation.ALTERNATIVE
    return CandidateRelation.COMPATIBLE_SET_MEMBER


def _merge_duplicate(winner: Candidate, duplicate: Candidate) -> None:
    for name in (
        "evidence_ids",
        "graph_evidence_ids",
        "spatial_support_ids",
        "independent_source_keys",
        "supporting_event_ids",
    ):
        values = getattr(winner, name)
        for value in getattr(duplicate, name):
            if value not in values:
                values.append(value)
    for value in duplicate.provenance:
        if value not in winner.provenance:
            winner.provenance.append(value)
    winner.origin = {**duplicate.origin, **winner.origin}
    winner.structural_signature = {**duplicate.structural_signature, **winner.structural_signature}
    winner.score = max(winner.score, duplicate.score)
    winner.activation = max(winner.activation, duplicate.activation)


def analyze_candidate_compatibility(
    workspace: BoundedAssociativeWorkspace,
) -> tuple[list[CandidateCompatibility], list[list[Candidate]]]:
    candidates = list(workspace.candidates)
    relations: list[CandidateCompatibility] = []
    duplicate_ids: set[str] = set()
    for index, left in enumerate(candidates):
        for right in candidates[index + 1:]:
            relation = classify_candidate_relation(left, right, workspace)
            relation_record = CandidateCompatibility(
                left_candidate_id=left.candidate_id,
                right_candidate_id=right.candidate_id,
                relation=relation,
                reason=relation.value.casefold(),
                confidence=1.0 if relation in {CandidateRelation.DUPLICATE, CandidateRelation.CONTRADICTORY} else 0.8,
                structural_match={
                    "same_gap": left.gap_id == right.gap_id,
                    "same_element": left.element_id == right.element_id,
                    "same_semantic_key": _semantic_key(left) == _semantic_key(right),
                },
            )
            relations.append(relation_record)
            if relation == CandidateRelation.DUPLICATE:
                winner, duplicate = sorted((left, right), key=lambda item: (-item.score, item.candidate_id))
                _merge_duplicate(winner, duplicate)
                duplicate_ids.add(duplicate.candidate_id)
            elif relation == CandidateRelation.CONTRADICTORY:
                conflict_id = _id("candidate-conflict", left.candidate_id, right.candidate_id)
                if not any(item.conflict_id == conflict_id for item in workspace.conflicts):
                    workspace.conflicts.append(Conflict(
                        conflict_id=conflict_id,
                        subject_id=left.gap_id,
                        competing_ids=(left.element_id, right.element_id),
                        reason="CONTRADICTORY_CANDIDATES",
                        severity=0.9,
                        provenance=tuple([*left.provenance, *right.provenance]),
                    ))
    if duplicate_ids:
        workspace.candidates = [item for item in workspace.candidates if item.candidate_id not in duplicate_ids]
        candidates = list(workspace.candidates)

    by_gap: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_gap[candidate.gap_id].append(candidate)
    groups: list[list[Candidate]] = []
    for gap_id in sorted(by_gap):
        remaining = sorted(by_gap[gap_id], key=lambda item: (-item.score, item.candidate_id))
        while remaining:
            seed = remaining.pop(0)
            group = [seed]
            for candidate in list(remaining):
                relation = classify_candidate_relation(seed, candidate, workspace)
                if relation == CandidateRelation.COMPATIBLE_SET_MEMBER:
                    group.append(candidate)
                    remaining.remove(candidate)
            groups.append(group)
    workspace.candidate_relations = relations
    workspace.candidate_groups = [
        {
            "group_id": _id("candidate-group", workspace.query_id, *[item.candidate_id for item in group]),
            "gap_id": group[0].gap_id,
            "candidate_ids": [item.candidate_id for item in group],
            "relation": "COMPATIBLE_SET_MEMBER" if len(group) > 1 else "SINGLETON",
        }
        for group in groups
    ]
    return relations, groups


def group_candidates(workspace: BoundedAssociativeWorkspace) -> list[list[Candidate]]:
    return analyze_candidate_compatibility(workspace)[1]


__all__ = [
    "classify_candidate_relation",
    "analyze_candidate_compatibility",
    "group_candidates",
]
