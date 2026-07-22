"""Candidate building, joint hypotheses and deterministic local resonance."""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping

from .contracts import AnswerStructure, BoundedAssociativeWorkspace, Candidate, Hypothesis, _id, clamp

DEFAULT_WEIGHTS = {
    "query": 0.20, "activation": 0.12, "event": 0.17, "context": 0.12,
    "type": 0.08, "gap": 0.12, "provenance": 0.12, "support": 0.07,
    "conflict": 0.20, "exclusion": 0.35, "redundancy": 0.08, "distance": 0.06,
}


def _norm(value: Any) -> str:
    return str(value or "").casefold().replace("ё", "е").strip()


def _values(element: Any) -> set[str]:
    payload = getattr(element, "payload", {}) or {}
    return {_norm(payload.get(key)) for key in ("value", "surface", "lemma", "head_lemma", "head_surface", "element_id") if payload.get(key)}


def _participant_values(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [item for item in record.get("participants") or () if isinstance(item, Mapping)]


def _option_values(option: Mapping[str, Any]) -> set[str]:
    return {
        _norm(option.get(key))
        for key in ("entity_id", "id", "element_id", "value", "surface", "lemma", "head_lemma", "head_surface")
        if option.get(key)
    }


def _same_value(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right or left in right or right in left:
        return True
    suffixes = ("ами", "ями", "ого", "ему", "ыми", "ов", "ев", "ом", "ем", "а", "я", "у", "ю", "е", "и", "ы", "ой", "ий", "ый", "ть", "л", "ли")
    def stem(value: str) -> str:
        suffix = next((item for item in suffixes if value.endswith(item) and len(value) - len(item) >= 3), "")
        return value[:-len(suffix)] if suffix else value
    return stem(left) == stem(right)


def _features(option: Mapping[str, Any]) -> Mapping[str, Any]:
    value = option.get("features") or option.get("feature_map") or option.get("features_json") or {}
    return value if isinstance(value, Mapping) else {}


def _constraint_fit(gap: Any, option: Mapping[str, Any], payload: Mapping[str, Any]) -> tuple[float, list[str]]:
    violations: list[str] = []
    features = _features(option)
    for constraint in gap.constraints:
        if not isinstance(constraint, Mapping) or constraint.get("type") != "grammatical_feature":
            continue
        feature = str(constraint.get("feature") or "")
        expected = _norm(constraint.get("value"))
        observed_values = {_norm(features.get(feature))}
        for alternative in features.get("morphology_alternatives") or ():
            if isinstance(alternative, Mapping):
                observed_values.add(_norm(alternative.get(feature)))
        if expected and expected not in observed_values:
            violations.append(f"GRAMMATICAL_FEATURE:{feature}:{expected}")
    relation = _norm(gap.expected_relation)
    if relation and relation != "predicate_attachment":
        observed_relations = {
            _norm(option.get("preposition")),
            _norm(option.get("relation")),
            _norm(payload.get("relation")),
            _norm(payload.get("predicate_relation")),
            _norm(payload.get("preposition")),
        }
        if relation not in observed_relations:
            violations.append(f"RELATION:{relation}")
    return (1.0 if not violations else 0.0), violations


def _initial_score(candidate: Candidate) -> float:
    return _candidate_score(candidate, DEFAULT_WEIGHTS)


def build_candidates(workspace: BoundedAssociativeWorkspace) -> BoundedAssociativeWorkspace:
    for gap in workspace.gaps:
        for element in list(workspace.events) + list(workspace.entities):
            if not element.evidence_ids or not element.provenance:
                continue
            payload = element.payload
            if "bee_result" in element.workspace_functions and not any(payload.get(key) for key in ("surface", "lemma", "value")):
                continue
            participants = _participant_values(payload) if element.element_type == "event" else []
            if element.element_type == "event" and not participants:
                continue
            options = participants or [payload]
            for ordinal, option in enumerate(options):
                candidate_element_id = str(option.get("entity_id") or option.get("element_id") or option.get("id") or option.get("lemma") or option.get("surface") or element.element_id)
                option_values = _option_values(option)
                excluded_values = {_norm(item) for item in (*gap.exclusions, *workspace.exclusions)}
                known_values = {_norm(item) for item in gap.known_elements}
                if (
                    not candidate_element_id
                    or any(_same_value(value, excluded) for value in option_values for excluded in excluded_values)
                    or any(_same_value(value, known) for value in option_values for known in known_values)
                ):
                    continue
                terms = _option_values(option)
                event_id = str(payload.get("event_id") or element.element_id)
                query_match = min(1.0, 0.30 + 0.35 * element.activation + 0.20 * bool(payload.get("predicate_lemma") or payload.get("predicate")) + 0.15 * bool(element.element_type == "event"))
                surface = str(option.get("surface") or option.get("head_surface") or option.get("value") or option.get("lemma") or candidate_element_id)
                lemma = str(option.get("lemma") or option.get("head_lemma") or surface)
                is_quantity = surface.replace(".", "", 1).isdigit()
                type_fit = 1.0 if gap.expected_type in {"unknown", "entity"} else (1.0 if gap.expected_type == "quantity" and is_quantity else 0.0)
                event_fit = 0.9 if element.element_type == "event" else 0.42
                context_fit = 0.85 if element.element_type == "context" else 0.4
                constraint_fit, constraint_violations = _constraint_fit(gap, option, payload)
                gap_fit = constraint_fit
                provenance_score = min(1.0, 0.35 + 0.15 * len(element.provenance) + 0.10 * len(element.retrieval_paths))
                conflict_score = min(1.0, 0.22 * len(element.conflict_ids))
                candidate_id = _id("candidate", gap.gap_id, candidate_element_id)
                candidate = Candidate(
                    candidate_id=candidate_id, gap_id=gap.gap_id, element_id=candidate_element_id,
                    activation=element.activation, query_match=query_match, event_fit=event_fit,
                    context_fit=context_fit, type_fit=type_fit, gap_fit=gap_fit,
                    provenance_score=provenance_score, conflict_score=conflict_score,
                    exclusion_score=0.0, semantic_distance=0.0 if terms else 0.7,
                    constraint_fit=constraint_fit, constraint_violations=constraint_violations,
                    evidence_ids=list(element.evidence_ids), provenance=list(element.provenance),
                    event_id=event_id, supporting_event_ids=[event_id], surface=surface, lemma=lemma,
                )
                candidate.score = _initial_score(candidate)
                workspace.add_candidate(candidate)
    return workspace


def build_hypotheses(workspace: BoundedAssociativeWorkspace) -> BoundedAssociativeWorkspace:
    by_gap: dict[str, list[Candidate]] = {}
    for candidate in workspace.candidates:
        by_gap.setdefault(candidate.gap_id, []).append(candidate)
    gaps = [gap.gap_id for gap in workspace.gaps]
    if not gaps:
        return workspace
    combinations = product(*(sorted(
        (item for item in by_gap.get(gap_id, []) if item.constraint_fit >= 1.0 and item.evidence_ids and item.provenance),
        key=lambda item: (-item.score, item.candidate_id),
    )[:workspace.budget.max_candidates_per_gap] for gap_id in gaps))
    for combination in combinations:
        if not combination:
            continue
        if len({item.candidate_id for item in combination}) != len(combination):
            continue
        event_sets = [set(item.supporting_event_ids or ([item.event_id] if item.event_id else ())) for item in combination]
        shared_events = set.intersection(*event_sets) if event_sets else set()
        if len(gaps) > 1 and not shared_events:
            continue
        fills = {item.gap_id: item.element_id for item in combination}
        evidence = list(dict.fromkeys(evidence_id for item in combination for evidence_id in item.evidence_ids))
        provenance = []
        for item in combination:
            for value in item.provenance:
                if value not in provenance:
                    provenance.append(value)
        hypothesis = Hypothesis(
            hypothesis_id=_id("hypothesis", workspace.query_id, "|".join(f"{key}:{value}" for key, value in sorted(fills.items()))),
            fills=fills, supporting_events=sorted(shared_events) if len(gaps) > 1 else sorted(event_sets[0]),
            candidate_ids=[item.candidate_id for item in combination],
            supporting_scenes=sorted({str(item.get("scene_id")) for item in provenance if item.get("scene_id")} ),
            evidence_ids=evidence[:workspace.budget.max_evidence_per_hypothesis], provenance=provenance,
            score=sum(item.score for item in combination) / len(combination),
        )
        if not any(item.hypothesis_id == hypothesis.hypothesis_id for item in workspace.hypotheses):
            workspace.hypotheses.append(hypothesis)
    workspace.hypotheses.sort(key=lambda item: (-item.score, item.hypothesis_id))
    workspace.hypotheses = workspace.hypotheses[:workspace.budget.max_hypotheses]
    return workspace


def _candidate_score(candidate: Candidate, weights: Mapping[str, float]) -> float:
    return clamp(
        weights["query"] * candidate.query_match
        + weights["activation"] * candidate.activation
        + weights["event"] * candidate.event_fit
        + weights["context"] * candidate.context_fit
        + weights["type"] * candidate.type_fit
        + weights["gap"] * candidate.gap_fit
        + weights["provenance"] * candidate.provenance_score
        + weights["support"] * candidate.mutual_support
        - weights["conflict"] * candidate.conflict_score
        - weights["exclusion"] * candidate.exclusion_score
        - weights["redundancy"] * candidate.redundancy_score
        - weights["distance"] * candidate.semantic_distance
    )


def run_resonance(workspace: BoundedAssociativeWorkspace, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = {"min_iterations": 1, "max_iterations": 5, "stable_iterations_required": 2, "answer_threshold": 0.68, "leader_margin": 0.12, "critical_conflict_threshold": 0.60, "weights": DEFAULT_WEIGHTS}
    cfg.update(dict(config or {}))
    weights = {**DEFAULT_WEIGHTS, **dict(cfg.get("weights") or {})}
    snapshots: list[dict[str, Any]] = []
    prior_leader = None
    stable_count = 0
    leader = None
    for iteration in range(1, int(cfg["max_iterations"]) + 1):
        for candidate in workspace.candidates:
            same_gap = [item for item in workspace.candidates if item.gap_id == candidate.gap_id and item.element_id == candidate.element_id]
            candidate.redundancy_score = max(0.0, min(1.0, (len(same_gap) - 1) * 0.1))
            support_count = sum(1 for item in workspace.candidates if item.event_id == candidate.event_id and item.element_id != candidate.element_id)
            candidate.mutual_support = min(1.0, 0.25 * support_count)
            candidate.score = _candidate_score(candidate, weights)
            if candidate.exclusion_score >= 0.5:
                candidate.status = "SUPPRESSED_EXCLUSION"
            elif candidate.constraint_fit < 1.0:
                candidate.status = "REJECTED_CONSTRAINT"
            elif candidate.conflict_score >= float(cfg["critical_conflict_threshold"]):
                candidate.status = "CONFLICTING"
            elif not candidate.evidence_ids or not candidate.provenance:
                candidate.status = "UNSUPPORTED"
            else:
                candidate.status = "ACTIVE"
        candidate_by_id = {item.candidate_id: item for item in workspace.candidates}
        for hypothesis in workspace.hypotheses:
            values = [candidate_by_id[item_id] for item_id in hypothesis.candidate_ids if item_id in candidate_by_id]
            hypothesis.score = sum(item.score for item in values) / len(values) if values else 0.0
            hypothesis.status = "ACTIVE" if len(values) == len(hypothesis.candidate_ids) and all(item.status == "ACTIVE" for item in values) else "CONFLICTING"
        workspace.hypotheses.sort(key=lambda item: (-item.score, item.hypothesis_id))
        active_hypotheses = [item for item in workspace.hypotheses if item.status == "ACTIVE"]
        leader = active_hypotheses[0] if active_hypotheses else None
        current_id = leader.hypothesis_id if leader else None
        stable_count = stable_count + 1 if current_id == prior_leader and current_id else 1
        prior_leader = current_id
        snapshot = {"stage": f"RESONANCE_{iteration}", "iteration": iteration, "leader_id": current_id, "leader_score": leader.score if leader else 0.0, "candidate_scores": {item.candidate_id: item.score for item in workspace.candidates}}
        snapshots.append(snapshot)
        if iteration >= int(cfg["min_iterations"]) and leader and stable_count >= int(cfg["stable_iterations_required"]):
            ranked = active_hypotheses
            margin = leader.score - (ranked[1].score if len(ranked) > 1 else 0.0)
            if leader.score >= float(cfg["answer_threshold"]) and margin >= float(cfg["leader_margin"]):
                break
    workspace.snapshots.extend(snapshots)
    workspace.resonance_state.update({"iteration": len(snapshots), "stability": clamp(stable_count / max(1, int(cfg["stable_iterations_required"]))), "leader_id": leader.hypothesis_id if leader else None})
    critical_conflict = any(item.severity >= float(cfg["critical_conflict_threshold"]) for item in workspace.conflicts)
    active_hypotheses = [item for item in workspace.hypotheses if item.status == "ACTIVE"]
    if critical_conflict:
        workspace.status = "CONFLICTING_EVIDENCE"
    elif not active_hypotheses:
        workspace.status = "PARTIAL_GAP_COMPLETION" if workspace.candidates and len(workspace.gaps) > 1 else ("NO_CANDIDATES" if not workspace.candidates else "INSUFFICIENT_EVIDENCE")
    else:
        nearest = active_hypotheses[1].score if len(active_hypotheses) > 1 else 0.0
        margin = leader.score - nearest if leader else 0.0
        if leader and leader.score >= float(cfg["answer_threshold"]) and margin >= float(cfg["leader_margin"]) and stable_count >= int(cfg["stable_iterations_required"]):
            workspace.status = "STABLE"
        elif leader and leader.score >= float(cfg["answer_threshold"]) and margin < float(cfg["leader_margin"]):
            workspace.status = "AMBIGUOUS_RESULT"
        else:
            workspace.status = "INSUFFICIENT_EVIDENCE"
    return {"status": workspace.status, "leader": leader.as_dict() if leader else None, "iterations": len(snapshots), "snapshots": snapshots, "stable": workspace.status == "STABLE"}


def should_dispatch_bees(workspace: BoundedAssociativeWorkspace, resonance_result: Mapping[str, Any]) -> dict[str, Any]:
    leader = resonance_result.get("leader") or {}
    score = float(leader.get("score") or 0.0)
    reasons = []
    if not workspace.candidates or workspace.status == "NO_CANDIDATES":
        reasons.append("GAP_WITHOUT_CANDIDATES")
    if workspace.status in {"INSUFFICIENT_EVIDENCE", "PARTIAL_GAP_COMPLETION", "CONFLICTING_EVIDENCE", "UNRESOLVED_CONTEXT"}:
        reasons.append(workspace.status)
    if score < 0.68:
        reasons.append("LOW_CONFIDENCE")
    if workspace.status == "AMBIGUOUS_RESULT":
        reasons.append("CLOSE_COMPETITORS")
    if workspace.status == "CONFLICTING_EVIDENCE":
        task_type = "FIND_CONTRADICTING_EVIDENCE"
    elif not workspace.candidates or workspace.status in {"NO_CANDIDATES", "PARTIAL_GAP_COMPLETION"}:
        task_type = "FIND_GAP_FILL"
    else:
        task_type = "FIND_SUPPORTING_EVIDENCE"
    return {"dispatch": bool(reasons), "reasons": list(dict.fromkeys(reasons)), "task_types": [task_type] if reasons else [], "bee_count": min(4, max(1, len(workspace.gaps))) if reasons else 0}


def compile_answer_structure(workspace: BoundedAssociativeWorkspace) -> AnswerStructure:
    leader = next((item for item in workspace.hypotheses if item.hypothesis_id == workspace.resonance_state.get("leader_id") and item.status == "ACTIVE"), None)
    if workspace.status == "STABLE" and leader:
        status = "STABLE"
        confidence = clamp(leader.score)
        fills = dict(leader.fills)
    elif workspace.status in {"AMBIGUOUS_RESULT", "UNRESOLVED_CONTEXT", "CONFLICTING_EVIDENCE", "NO_CANDIDATES", "PARTIAL_GAP_COMPLETION"}:
        status, confidence, fills = workspace.status, clamp(leader.score if leader else 0.0), {}
    else:
        status, confidence, fills = "INSUFFICIENT_EVIDENCE", 0.0, {}
    rejected = tuple({"gap_id": item.gap_id, "candidate": item.element_id, "reason": item.status} for item in workspace.candidates if item.status != "ACTIVE")
    evidence = tuple(leader.evidence_ids if leader else ())
    candidate_by_id = {item.candidate_id: item for item in workspace.candidates}
    surfaces = {
        candidate_by_id[candidate_id].gap_id: candidate_by_id[candidate_id].surface or candidate_by_id[candidate_id].element_id
        for candidate_id in leader.candidate_ids
        if candidate_id in candidate_by_id
    } if leader and status == "STABLE" else {}
    provenance = {"query_id": workspace.query_id, "workspace_id": workspace.workspace_id, "hypothesis_id": leader.hypothesis_id if leader else None, "evidence_ids": list(evidence), "resonance_iteration": workspace.resonance_state.get("iteration", 0), "surface_forms": surfaces}
    return AnswerStructure("multi_gap" if len(fills) > 1 else "entity", status, fills, confidence, evidence, rejected, tuple([] if status == "STABLE" else [status]), {"language": "ru", "do_not_invent": True, "state_uncertainty": status != "STABLE"}, provenance)


def render_answer(answer_structure: AnswerStructure, language_config: Mapping[str, Any] | None = None) -> str:
    if answer_structure.status != "STABLE":
        return answer_structure.status
    surfaces = answer_structure.provenance.get("surface_forms") if isinstance(answer_structure.provenance, Mapping) else {}
    values = [str((surfaces or {}).get(gap_id) or value) for gap_id, value in answer_structure.filled_gaps.items()]
    return ", ".join(values) if values else "INSUFFICIENT_EVIDENCE"
