"""Candidate building, joint hypotheses and deterministic local resonance."""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping

from .contracts import (
    AnswerStructure,
    BoundedAssociativeWorkspace,
    Candidate,
    EventCandidateConfiguration,
    Hypothesis,
    _id,
    clamp,
)

DEFAULT_WEIGHTS = {
    "query": 0.20, "activation": 0.12, "event": 0.17, "context": 0.12,
    "type": 0.08, "gap": 0.12, "provenance": 0.12, "support": 0.07, "field": 0.08,
    "conflict": 0.20, "exclusion": 0.35, "redundancy": 0.08, "distance": 0.06,
}


def _norm(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("lemma") or value.get("surface") or ""
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
    return bool(left and right and left == right)


def _features(option: Mapping[str, Any]) -> Mapping[str, Any]:
    value = option.get("features") or option.get("feature_map") or option.get("features_json") or {}
    return value if isinstance(value, Mapping) else {}


def _constraint_fit(gap: Any, option: Mapping[str, Any], payload: Mapping[str, Any]) -> tuple[float, list[str]]:
    violations: list[str] = []
    hard_violation = False
    alternatives: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    features = _features(option)
    for constraint in gap.constraints:
        if not isinstance(constraint, Mapping) or str(constraint.get("type") or "").upper() != "GRAMMATICAL_FEATURE":
            continue
        feature = str(constraint.get("feature") or "")
        expected = _norm(constraint.get("value"))
        alternatives.setdefault(feature, []).append((expected, constraint))
        observed_values = {_norm(features.get(feature))}
        for alternative in features.get("morphology_alternatives") or ():
            if isinstance(alternative, Mapping):
                observed_values.add(_norm(alternative.get(feature)))
    for feature, values in alternatives.items():
        observed_values = {_norm(features.get(feature))}
        for alternative in features.get("morphology_alternatives") or ():
            if isinstance(alternative, Mapping):
                observed_values.add(_norm(alternative.get(feature)))
        expected_values = {expected for expected, _ in values if expected}
        if expected_values and not expected_values & observed_values:
            violations.append(f"GRAMMATICAL_FEATURE:{feature}:{'|'.join(sorted(expected_values))}")
            hard_violation = hard_violation or any(str(item.get("hardness") or "SOFT").upper() == "HARD" for _, item in values)
    return (0.0 if hard_violation else (1.0 if not violations else 0.82)), violations


def _gap_hint_fit(gap: Any, option: Mapping[str, Any]) -> float:
    """Use interrogatives as ranking hints, never as mandatory case rules."""
    focus = _norm(getattr(gap, "surface_projection", ""))
    features = _features(option)
    animacy = _norm(features.get("animacy"))
    grammatical_case = _norm(features.get("case"))
    preposition = _norm(option.get("preposition") or features.get("preposition_support"))
    if focus in {"кто", "кого", "кому", "кем", "ком"}:
        fit = 0.80 if animacy == "anim" else 0.20
        expected_case = {"кто": "nomn", "кого": "gent", "кому": "datv", "кем": "ablt", "ком": "loct"}.get(focus)
        if expected_case:
            return 1.0 if animacy == "anim" and grammatical_case == expected_case else fit * 0.70
        return min(1.0, fit + 0.20)
    if focus in {"что", "чего", "чему", "чем", "о чем"}:
        fit = 1.0 if animacy == "inan" else 0.55
        expected_case = {"что": "accs", "чего": "gent", "чему": "datv", "чем": "ablt", "о чем": "loct"}.get(focus)
        return min(1.0, fit + (0.18 if expected_case and grammatical_case == expected_case else 0.0))
    if focus in {"где", "куда", "откуда"}:
        return 1.0 if preposition or grammatical_case in {"loct", "datv", "gent"} else 0.45
    return 1.0


def _record_values(record: Mapping[str, Any]) -> set[str]:
    return _option_values(record)


def _matches_known(known: str, participant: Mapping[str, Any]) -> bool:
    return any(_same_value(_norm(known), value) for value in _record_values(participant))


def _relation_values(record: Mapping[str, Any], payload: Mapping[str, Any]) -> set[str]:
    values = {
        _norm(record.get(key))
        for key in ("preposition", "relation", "relation_value", "predicate_relation")
        if record.get(key)
    }
    values |= {
        _norm(payload.get(key))
        for key in ("preposition", "relation", "relation_value", "predicate_relation")
        if payload.get(key)
    }
    return values


def _relation_satisfied(expected: str, participant: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    expected = _norm(expected)
    if not expected or expected == "predicate_attachment":
        return True
    observed = _relation_values(participant, payload)
    return expected in observed


def _workspace_predicate_lemmas(workspace: BoundedAssociativeWorkspace) -> set[str]:
    values = {_norm(workspace.explicit_predicate)} if workspace.explicit_predicate else set()
    for item in workspace.predicate_hypotheses:
        if isinstance(item, Mapping):
            value = _norm(item.get("predicate") or item.get("lemma"))
            if value:
                values.add(value)
    return {item for item in values if item}


def _attachment_option(participant: Mapping[str, Any]) -> dict[str, Any]:
    option = dict(participant)
    base_surface = str(
        participant.get("surface")
        or participant.get("head_surface")
        or participant.get("value")
        or participant.get("head_lemma")
        or ""
    ).strip()
    preposition = str(participant.get("preposition") or "").strip()
    option["surface"] = f"{preposition} {base_surface}".strip()
    option["lemma"] = str(participant.get("head_lemma") or participant.get("lemma") or base_surface)
    option["relation_surface"] = preposition or None
    option["attachment_type_hypotheses"] = [
        {"type": "PREPOSITIONAL_ATTACHMENT", "confidence": 0.82}
    ] if preposition else [{"type": "EVENT_PARTICIPANT", "confidence": 0.55}]
    return option


def build_configurations(workspace: BoundedAssociativeWorkspace) -> BoundedAssociativeWorkspace:
    """Turn retrieved events into the only admissible evidence structures."""
    workspace.configurations.clear()
    known_values: list[Any] = []
    known_keys: set[str] = set()
    for gap in workspace.gaps:
        for value in gap.known_elements:
            key = _norm(value)
            if key and key not in known_keys:
                known_keys.add(key)
                known_values.append(value)
    known = tuple(known_values)
    for event in workspace.events:
        payload = event.payload
        participants = [item for item in payload.get("participants") or () if isinstance(item, Mapping)]
        if not participants:
            continue
        bindings: list[Mapping[str, Any]] = []
        used: set[str] = set()
        reasons: list[str] = []
        for value in known:
            participant = next((item for item in participants if _matches_known(value, item)), None)
            if participant is None:
                reasons.append("KNOWN_ELEMENT_NOT_FOUND")
                continue
            participant_id = str(participant.get("entity_id") or participant.get("element_id") or participant.get("id") or "")
            if participant_id in used:
                reasons.append("KNOWN_MEMBER_BINDING_CONFLICT")
                continue
            used.add(participant_id)
            relation = next((gap.expected_relation for gap in workspace.gaps if value in gap.known_elements and gap.expected_relation), None)
            relation_ok = _relation_satisfied(str(relation or ""), participant, payload)
            bindings.append({"known": value, "participant_id": participant_id})
            if relation and not relation_ok:
                reasons.append("RELATION_NOT_MATCHED")
        event_id = str(payload.get("event_id") or event.element_id)
        polarity = _norm(payload.get("polarity") or payload.get("negation") or "positive")
        query_is_negative = any(
            str(item.get("type") or "").upper() == "NEGATION"
            for item in workspace.constraints if isinstance(item, Mapping)
        )
        if polarity in {"negative", "negated", "false"} and not query_is_negative:
            reasons.append("NEGATION_CONFLICT")
        if query_is_negative and polarity not in {"negative", "negated", "false"}:
            reasons.append("NEGATION_CONFLICT")
        temporal_match: Mapping[str, Any] | None = None
        if workspace.temporal_scope:
            requested = str(workspace.temporal_scope.get("kind") or "CURRENT").upper()
            actuality = str(payload.get("actuality") or payload.get("state_status") or "ACTUAL").upper()
            superseded = bool(payload.get("superseded") or payload.get("is_superseded") or actuality in {"SUPERSEDED", "ENDED", "RETRACTED"})
            compatible = not (requested == "CURRENT" and superseded)
            temporal_match = {"scope": requested, "actuality": actuality, "status": "SATISFIED" if compatible else "HARD_CONFLICT"}
            if not compatible:
                reasons.append("TEMPORAL_CONFLICT")
        configuration_id = _id("configuration", workspace.query_id, event_id, "|".join(sorted(used)))
        predicate = payload.get("predicate_lemma") or payload.get("predicate") or payload.get("action")
        expected_predicates = _workspace_predicate_lemmas(workspace)
        predicate_match = not expected_predicates or _norm(predicate) in expected_predicates
        if not predicate_match:
            reasons.append("PREDICATE_MISMATCH")
        predicate_status = "COMPATIBLE" if predicate_match and predicate else ("UNRESOLVED" if not predicate else "HARD_CONFLICT")
        configuration = EventCandidateConfiguration(
            configuration_id=configuration_id,
            query_id=workspace.query_id,
            event_id=event_id,
            known_member_bindings=list(bindings),
            predicate_binding={"value": predicate, "status": predicate_status, "hypotheses": sorted(expected_predicates)} if predicate else None,
            relation_matches=[item for item in bindings if item.get("known")],
            temporal_match=temporal_match,
            constraint_evaluations=[
                {"status": "SATISFIED" if not reasons else "HARD_CONFLICT", "reason": reason}
                for reason in reasons
            ],
            evidence_ids=list(event.evidence_ids),
            conflict_ids=list(event.conflict_ids),
            score=clamp(event.activation * (1.0 if not reasons else 0.0)),
            status="REJECTED" if reasons else "ACTIVE",
            rejection_reasons=list(dict.fromkeys(reasons)),
            graph_fit=clamp(event.activation),
            field_fit=clamp(sum(1.0 for support in workspace.spatial_support if any(str(participant.get("entity_id") or participant.get("id") or "") == str(next((cloud.payload.get("concept_id") for cloud in workspace.clouds if cloud.element_id == support.cloud_id), "")) for participant in participants)) / max(1, len(participants))),
            gradient_alignment=clamp(sum(float(item.get("weight") or 0.0) for item in workspace.local_gradients) / max(1, len(workspace.local_gradients))),
            cloud_overlap=clamp(len(workspace.clouds) / max(1, len(participants))),
            semantic_region_id=f"region:{workspace.query_id}",
            known_bindings_by_gap={gap.gap_id: [item for item in bindings if item.get("known") in gap.known_elements] for gap in workspace.gaps},
            relation_evaluations=[{"type": "RELATION", "target": item.get("participant_id"), "status": "EVALUATED"} for item in bindings],
            predicate_evaluation={"value": predicate, "status": predicate_status, "hypotheses": sorted(expected_predicates)},
            temporal_evaluation=dict(temporal_match or {}),
            polarity_evaluation={"value": polarity, "status": "COMPATIBLE" if "NEGATION_CONFLICT" not in reasons else "HARD_CONFLICT"},
            state_evaluation={"status": "UNRESOLVED"},
        )
        workspace.configurations.append(configuration)
    workspace.configurations.sort(key=lambda item: (-item.score, item.configuration_id))
    return workspace


def _initial_score(candidate: Candidate) -> float:
    return _candidate_score(candidate, DEFAULT_WEIGHTS)


def build_candidates(workspace: BoundedAssociativeWorkspace) -> BoundedAssociativeWorkspace:
    if not workspace.configurations:
        build_configurations(workspace)
    for gap in workspace.gaps:
        if workspace.query_type == "associative_question":
            for cloud in workspace.clouds:
                support_ids = [str(item.support_id) for item in workspace.spatial_support if item.cloud_id == cloud.element_id]
                if not support_ids:
                    continue
                payload = cloud.payload
                candidate = Candidate(
                    candidate_id=_id("candidate", "field", gap.gap_id, cloud.element_id), gap_id=gap.gap_id,
                    element_id=cloud.element_id, activation=cloud.activation, query_match=0.86,
                    event_fit=0.0, context_fit=0.72, type_fit=1.0, gap_fit=1.0,
                    field_fit=0.92, provenance_score=0.35, semantic_distance=0.0,
                    evidence_ids=[], spatial_support_ids=support_ids, provenance=list(cloud.provenance),
                    supporting_event_ids=[], surface=str(payload.get("concept_id") or cloud.element_id),
                    lemma=str(payload.get("concept_id") or cloud.element_id),
                    origin={"type": "FIELD_NEIGHBOURHOOD", "cloud_id": cloud.element_id},
                )
                candidate.score = _initial_score(candidate)
                workspace.add_candidate(candidate)
        for configuration in workspace.configurations:
            if configuration.status != "ACTIVE":
                continue
            element = next((item for item in workspace.events if item.element_id == configuration.event_id), None)
            if element is None or not element.evidence_ids or not element.provenance:
                continue
            payload = element.payload
            if "bee_result" in element.workspace_functions and not any(payload.get(key) for key in ("surface", "lemma", "value")):
                continue
            participants = _participant_values(payload)
            predicate_option = {
                "element_id": f"predicate:{payload.get('predicate_lemma') or payload.get('predicate') or payload.get('action') or element.element_id}",
                "surface": payload.get("predicate_surface") or payload.get("predicate") or payload.get("action"),
                "lemma": payload.get("predicate_lemma") or payload.get("predicate") or payload.get("action"),
                "features": payload.get("predicate_features") or {},
            }
            if gap.expected_type == "predicate":
                options = [predicate_option]
            elif gap.expected_type == "component":
                options = [
                    {**component, "element_id": component.get("component_id") or component.get("id")}
                    for participant in participants
                    for component in participant.get("components") or ()
                    if isinstance(component, Mapping)
                ]
            elif gap.expected_type in {"attachment", "entity", "location", "property", "time", "state", "unknown"}:
                options = [_attachment_option(participant) for participant in participants]
            else:
                options = []
            known_ids = {str(item.get("participant_id") or "") for item in configuration.known_member_bindings}
            for ordinal, option in enumerate(options):
                candidate_element_id = str(option.get("entity_id") or option.get("element_id") or option.get("id") or option.get("lemma") or option.get("surface") or element.element_id)
                if candidate_element_id in known_ids:
                    continue
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
                query_match = max(query_match, 0.65 + 0.10 * bool(configuration.known_member_bindings))
                surface = str(option.get("surface") or option.get("head_surface") or option.get("value") or option.get("lemma") or candidate_element_id)
                lemma = str(option.get("lemma") or option.get("head_lemma") or surface)
                is_quantity = surface.replace(".", "", 1).isdigit()
                type_fit = 1.0 if gap.expected_type in {"unknown", "entity", "attachment", "predicate", "component", "property", "location", "time", "state"} else (1.0 if gap.expected_type == "quantity" and is_quantity else 0.0)
                event_fit = 1.0 if element.element_type == "event" else 0.42
                context_fit = 0.85 if element.element_type == "context" else 0.4
                constraint_fit, constraint_violations = _constraint_fit(gap, option, payload)
                gap_fit = min(1.0, constraint_fit * _gap_hint_fit(gap, option))
                provenance_score = min(1.0, 0.35 + 0.15 * len(element.provenance) + 0.10 * len(element.retrieval_paths))
                conflict_score = min(1.0, 0.22 * len(element.conflict_ids))
                candidate_id = _id("candidate", configuration.configuration_id, gap.gap_id, candidate_element_id)
                candidate_spatial_support_ids = [
                    support.support_id
                    for support in workspace.spatial_support
                    if any(
                        cloud.element_id == support.cloud_id
                        and str(cloud.payload.get("concept_id") or "") == candidate_element_id
                        for cloud in workspace.clouds
                    )
                ]
                candidate = Candidate(
                    candidate_id=candidate_id, gap_id=gap.gap_id, element_id=candidate_element_id,
                    activation=element.activation, query_match=query_match, event_fit=event_fit,
                    context_fit=context_fit, type_fit=type_fit, gap_fit=gap_fit,
                    field_fit=configuration.field_fit,
                    provenance_score=provenance_score, conflict_score=conflict_score,
                    exclusion_score=0.0, semantic_distance=0.0 if terms else 0.7,
                    constraint_fit=constraint_fit, constraint_violations=constraint_violations,
                    evidence_ids=list(element.evidence_ids),
                    graph_evidence_ids=list(element.evidence_ids),
                    spatial_support_ids=list(dict.fromkeys(candidate_spatial_support_ids)),
                    independent_source_keys=list(dict.fromkeys(
                        str(item.get("independent_source_key"))
                        for item in element.provenance
                        if isinstance(item, Mapping) and item.get("independent_source_key")
                    )),
                    provenance=list(element.provenance),
                    event_id=event_id, supporting_event_ids=[event_id], surface=surface, lemma=lemma,
                    configuration_id=configuration.configuration_id,
                    origin={
                        "type": "EVENT_PREDICATE" if gap.expected_type == "predicate" else ("MENTION_COMPONENT" if gap.expected_type == "component" else ("EVENT_ATTACHMENT" if gap.expected_type == "attachment" else "EVENT_PARTICIPANT")), "event_id": event_id,
                        "configuration_id": configuration.configuration_id,
                        "source_element_id": candidate_element_id,
                    },
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
        (item for item in by_gap.get(gap_id, [])
         if item.constraint_fit >= (1.0 if len(gaps) > 1 else 0.0)
         and (item.evidence_ids or item.spatial_support_ids) and item.provenance),
        key=lambda item: (-item.score, item.candidate_id),
    )[:workspace.budget.max_candidates_per_gap] for gap_id in gaps))
    for combination in combinations:
        if not combination:
            continue
        if len({item.candidate_id for item in combination}) != len(combination):
            continue
        if len({item.element_id for item in combination}) != len(combination):
            continue
        event_sets = [set(item.supporting_event_ids or ([item.event_id] if item.event_id else ())) for item in combination]
        shared_events = set.intersection(*event_sets) if event_sets else set()
        configuration_ids = {item.configuration_id for item in combination if item.configuration_id}
        if len(gaps) > 1 and (not shared_events or len(configuration_ids) != 1):
            continue
        fills = {item.gap_id: item.element_id for item in combination}
        evidence = list(dict.fromkeys(evidence_id for item in combination for evidence_id in item.evidence_ids))
        spatial_support = list(dict.fromkeys(support_id for item in combination for support_id in item.spatial_support_ids))
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
            configuration_ids=sorted(configuration_ids),
            graph_evidence_ids=list(evidence),
            spatial_support_ids=spatial_support,
            active_cloud_ids=sorted({item.element_id for item in workspace.clouds if item.element_id in {candidate.element_id for candidate in combination}}),
            field_region_id=f"region:{workspace.query_id}",
            evidential_score=clamp(sum(item.provenance_score for item in combination) / len(combination)),
            spatial_score=clamp(sum(item.field_fit for item in combination) / len(combination)),
            joint_score=clamp(sum(item.score for item in combination) / len(combination)),
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
        + weights["field"] * candidate.field_fit
        - weights["conflict"] * candidate.conflict_score
        - weights["exclusion"] * candidate.exclusion_score
        - weights["redundancy"] * candidate.redundancy_score
        - weights["distance"] * candidate.semantic_distance
    )


def run_resonance(workspace: BoundedAssociativeWorkspace, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not workspace.hypotheses:
        workspace.resonance_state.update({"iteration": 0, "stability": 0.0, "leader_id": None, "stop_reason": "NO_HYPOTHESES"})
        workspace.status = "INSUFFICIENT_EVIDENCE"
        return {"status": workspace.status, "leader": None, "iterations": 0, "snapshots": [], "stable": False, "stop_reason": "NO_HYPOTHESES"}
    cfg = {"min_iterations": 1, "max_iterations": 5, "stable_iterations_required": 2, "answer_threshold": 0.68, "leader_margin": 0.05, "critical_conflict_threshold": 0.60, "weights": DEFAULT_WEIGHTS}
    cfg.update(dict(config or {}))
    if workspace.query_type == "associative_question":
        cfg["answer_threshold"] = min(float(cfg["answer_threshold"]), 0.55)
        cfg["leader_margin"] = 0.0
    elif len(workspace.gaps) > 1:
        cfg["leader_margin"] = min(float(cfg["leader_margin"]), 0.03)
    weights = {**DEFAULT_WEIGHTS, **dict(cfg.get("weights") or {})}
    snapshots: list[dict[str, Any]] = []
    prior_leader = None
    stable_count = 0
    leader = None
    for iteration in range(1, int(cfg["max_iterations"]) + 1):
        for candidate in workspace.candidates:
            same_gap = [item for item in workspace.candidates if item.gap_id == candidate.gap_id and item.element_id == candidate.element_id]
            candidate.redundancy_score = max(0.0, min(1.0, (len(same_gap) - 1) * 0.1))
            independent_sources = {
                str(evidence.independent_source_key or evidence.source_id)
                for evidence in workspace.evidence
                if evidence.evidence_id in candidate.evidence_ids
            }
            candidate.mutual_support = min(1.0, 0.25 * max(0, len(independent_sources) - 1))
            candidate.score = _candidate_score(candidate, weights)
            if candidate.exclusion_score >= 0.5:
                candidate.status = "SUPPRESSED_EXCLUSION"
            elif candidate.constraint_fit <= 0.0:
                candidate.status = "REJECTED_CONSTRAINT"
            elif candidate.conflict_score >= float(cfg["critical_conflict_threshold"]):
                candidate.status = "CONFLICTING"
            elif not candidate.evidence_ids and not candidate.spatial_support_ids or not candidate.provenance:
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
    if resonance_result.get("stop_reason") == "INTERNAL_PIPELINE_ERROR":
        return {"dispatch": False, "reasons": ["INTERNAL_PIPELINE_ERROR"], "task_types": [], "bee_count": 0}
    leader = resonance_result.get("leader") or {}
    score = float(leader.get("score") or 0.0)
    reasons = []
    if not workspace.candidates or workspace.status == "NO_CANDIDATES":
        reasons.append("GAP_WITHOUT_CANDIDATES")
    if workspace.status in {"INSUFFICIENT_EVIDENCE", "PARTIAL_GAP_COMPLETION", "CONFLICTING_EVIDENCE", "UNRESOLVED_CONTEXT", "NO_HYPOTHESES"}:
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
    elif workspace.status in {"AMBIGUOUS_RESULT", "UNRESOLVED_CONTEXT", "CONFLICTING_EVIDENCE", "NO_CANDIDATES", "NO_HYPOTHESES", "PARTIAL_GAP_COMPLETION"}:
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
    if leader and len(leader.supporting_events) == 1:
        event_id = leader.supporting_events[0]
        event = next((item for item in workspace.events if str(item.payload.get("event_id") or item.element_id) == event_id), None)
        if event is not None:
            event_surface = str(event.payload.get("source_surface") or event.payload.get("raw_text") or "").strip()
            if event_surface:
                provenance["event_surface"] = event_surface.rstrip(".?!")
    answer_type = "multi_gap" if len(fills) > 1 else "entity"
    if leader:
        candidate_by_id = {item.candidate_id: item for item in workspace.candidates}
        if any((candidate_by_id.get(item_id) is not None and candidate_by_id[item_id].origin.get("type") == "EVENT_PREDICATE") for item_id in leader.candidate_ids):
            answer_type = "predicate"
    graph_support = clamp(sum(candidate_by_id[item].provenance_score for item in leader.candidate_ids if item in candidate_by_id) / max(1, len(leader.candidate_ids))) if leader else 0.0
    field_support = clamp(sum(candidate_by_id[item].field_fit for item in leader.candidate_ids if item in candidate_by_id) / max(1, len(leader.candidate_ids))) if leader else 0.0
    graph_evidence = tuple(leader.graph_evidence_ids if leader else ())
    spatial_support = tuple(leader.spatial_support_ids if leader else ())
    independent_source_keys = {
        str(item.get("independent_source_key"))
        for item in (leader.provenance if leader else ())
        if item.get("independent_source_key")
    }
    independent_source_count = len(independent_source_keys)
    epistemic_mode = "OBSERVED" if status == "STABLE" and graph_evidence and independent_source_count >= 1 and not (leader.conflict_ids if leader else ()) else ("UNKNOWN" if status != "STABLE" else "INFERRED")
    if status == "STABLE" and workspace.query_type == "associative_question":
        epistemic_mode = "ASSOCIATIVE"
    if status == "STABLE" and not graph_evidence:
        epistemic_mode = "ASSOCIATIVE" if workspace.query_type == "associative_question" else "PREDICTED"
    return AnswerStructure(answer_type, status, fills, confidence, evidence, rejected, tuple([] if status == "STABLE" else [status]), {"language": "ru", "do_not_invent": True, "state_uncertainty": status != "STABLE"}, provenance, epistemic_mode, graph_support, field_support, graph_evidence, spatial_support, independent_source_count)


def render_answer(answer_structure: AnswerStructure, language_config: Mapping[str, Any] | None = None) -> str:
    if answer_structure.status != "STABLE":
        return answer_structure.status
    surfaces = answer_structure.provenance.get("surface_forms") if isinstance(answer_structure.provenance, Mapping) else {}
    values = [str((surfaces or {}).get(gap_id) or value) for gap_id, value in answer_structure.filled_gaps.items()]
    if answer_structure.answer_type == "multi_gap":
        event_surface = str(answer_structure.provenance.get("event_surface") or "")
        if event_surface:
            return event_surface
    return ", ".join(values) if values else "INSUFFICIENT_EVIDENCE"
