"""Deterministic multi-cycle interpretation with traceable evidence."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .evidence import EvidenceAggregator
from .models import (
    ClauseMode,
    DialogueActType,
    EvidencePacket,
    HypothesisStatus,
    INTERPRETATION_VERSION,
    InterpretationHypothesis,
    InterpretationStatus,
    LanguageAnalysis,
)


@dataclass(frozen=True)
class InterpretationConfig:
    version: str = "dialogue-v2.5-thresholds-1"
    max_interpretation_cycles: int = 3
    stable_cycles_required: int = 2
    confirmation_support: float = 0.68
    ambiguity_margin: float = 0.08
    ambiguity_support: float = 0.52
    conflict_support: float = 0.78


DEFAULT_INTERPRETATION_CONFIG = InterpretationConfig()


def _stable_id(prefix: str, *parts: object) -> str:
    key = "|".join(str(part) for part in parts)
    return f"{prefix}-{uuid.uuid5(uuid.NAMESPACE_URL, key).hex[:20]}"


def _canonical(value: Any) -> str:
    if hasattr(value, "value"):
        value = value.value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class InterpretationEngine:
    def __init__(
        self,
        config: Optional[InterpretationConfig] = None,
        aggregator: Optional[EvidenceAggregator] = None,
    ) -> None:
        self.config = config or DEFAULT_INTERPRETATION_CONFIG
        self.aggregator = aggregator or EvidenceAggregator()

    @staticmethod
    def _packet(
        hypothesis: InterpretationHypothesis,
        *,
        origin: str,
        value: Any,
        support: float,
        evidence_type: str,
        group: str,
        penalty: float = 0.0,
        token_start: Optional[int] = None,
        token_end: Optional[int] = None,
        source_object_id: Optional[str] = None,
    ) -> EvidencePacket:
        return EvidencePacket(
            id=_stable_id(
                "evidence",
                origin,
                hypothesis.scope_type,
                hypothesis.scope_id,
                hypothesis.id,
                evidence_type,
                token_start,
                token_end,
                INTERPRETATION_VERSION,
            ),
            origin=origin,
            target_hypothesis_id=hypothesis.id,
            value=value,
            support=support,
            penalty=penalty,
            evidence_type=evidence_type,
            independent_group=group,
            scope_type=hypothesis.scope_type,
            scope_id=hypothesis.scope_id,
            source_token_start=token_start,
            source_token_end=token_end,
            source_object_id=source_object_id,
        )

    def _build(
        self,
        analysis: LanguageAnalysis,
        reference_candidates: Optional[
            Mapping[int, Sequence[Dict[str, Any]]]
        ] = None,
    ) -> tuple[List[InterpretationHypothesis], List[EvidencePacket]]:
        hypotheses: List[InterpretationHypothesis] = []
        packets: List[EvidencePacket] = []
        utterance_id = (
            analysis.utterance.id if analysis.utterance else ""
        )
        for token in analysis.tokens:
            token_scope_id = (
                f"{utterance_id}:token:{token.index}"
                if utterance_id else str(token.index)
            )
            for variant_index, variant in enumerate(token.analyses):
                hypothesis = InterpretationHypothesis(
                    id=_stable_id(
                        "hypothesis",
                        utterance_id,
                        "token",
                        token.index,
                        "morphology",
                        variant_index,
                        variant.lemma,
                        variant.pos,
                    ),
                    scope_type="token",
                    scope_id=token_scope_id,
                    hypothesis_type="morphology",
                    value={
                        "lemma": variant.lemma,
                        "part_of_speech": variant.pos,
                        "features": dict(variant.features),
                    },
                )
                hypotheses.append(hypothesis)
                packets.append(self._packet(
                    hypothesis,
                    origin="morphology",
                    value=hypothesis.value,
                    support=variant.confidence,
                    evidence_type="morphological_parse",
                    group="morphology",
                    token_start=token.index,
                    token_end=token.index,
                ))
                for evidence in variant.evidence:
                    if evidence == "morphology_confidence":
                        continue
                    group = (
                        "agreement"
                        if "agreement" in evidence or "government" in evidence
                        else "syntax"
                    )
                    packets.append(self._packet(
                        hypothesis,
                        origin="language_analyzer",
                        value=hypothesis.value,
                        support=max(
                            0.18,
                            min(0.92, float(variant.confidence) + 0.12),
                        ),
                        evidence_type=evidence,
                        group=group,
                        token_start=token.index,
                        token_end=token.index,
                    ))
        for act in analysis.dialogue_acts:
            hypothesis = InterpretationHypothesis(
                id=_stable_id("hypothesis", act.id, "dialogue_act", act.act_type.value),
                scope_type="dialogue_act",
                scope_id=act.id,
                hypothesis_type="dialogue_act",
                value=act.act_type.value,
            )
            hypotheses.append(hypothesis)
            packets.append(self._packet(
                hypothesis,
                origin="dialogue_act_parser",
                value=act.act_type.value,
                support=act.confidence,
                evidence_type=(
                    str(act.evidence[0].get("type"))
                    if act.evidence else "act_construction"
                ),
                group="discourse",
                token_start=act.token_start,
                token_end=act.token_end,
                source_object_id=act.id,
            ))
            for index, alternative in enumerate(act.alternatives):
                value = alternative.get("act_type")
                alternative_hypothesis = InterpretationHypothesis(
                    id=_stable_id(
                        "hypothesis",
                        act.id,
                        "dialogue_act",
                        "alternative",
                        index,
                        value,
                    ),
                    scope_type="dialogue_act",
                    scope_id=act.id,
                    hypothesis_type="dialogue_act",
                    value=value,
                )
                hypotheses.append(alternative_hypothesis)
                packets.append(self._packet(
                    alternative_hypothesis,
                    origin="dialogue_act_parser",
                    value=value,
                    support=float(alternative.get("confidence", 0.0)),
                    evidence_type="act_alternative",
                    group="discourse",
                    token_start=act.token_start,
                    token_end=act.token_end,
                    source_object_id=act.id,
                ))
        for clause in analysis.clauses:
            boundary_hypothesis = InterpretationHypothesis(
                id=_stable_id(
                    "hypothesis",
                    clause.id,
                    "clause_boundary",
                    clause.token_start,
                    clause.token_end,
                ),
                scope_type="clause",
                scope_id=clause.id,
                hypothesis_type="clause_boundary",
                value={
                    "token_start": clause.token_start,
                    "token_end": clause.token_end,
                },
            )
            hypotheses.append(boundary_hypothesis)
            packets.append(self._packet(
                boundary_hypothesis,
                origin="clause_parser",
                value=boundary_hypothesis.value,
                support=0.9,
                evidence_type="predicate_and_punctuation_boundary",
                group="syntax",
                token_start=clause.token_start,
                token_end=clause.token_end,
                source_object_id=clause.id,
            ))
            for alternative in clause.alternative_boundaries:
                alternative_hypothesis = InterpretationHypothesis(
                    id=_stable_id(
                        "hypothesis",
                        clause.id,
                        "clause_boundary",
                        alternative.get("token_start"),
                        alternative.get("token_end"),
                    ),
                    scope_type="clause",
                    scope_id=clause.id,
                    hypothesis_type="clause_boundary",
                    value={
                        "token_start": alternative.get("token_start"),
                        "token_end": alternative.get("token_end"),
                    },
                )
                hypotheses.append(alternative_hypothesis)
                packets.append(self._packet(
                    alternative_hypothesis,
                    origin="clause_parser",
                    value=alternative_hypothesis.value,
                    support=float(alternative.get("confidence", 0.0)),
                    evidence_type=str(
                        alternative.get("reason")
                        or "alternative_clause_boundary"
                    ),
                    group="syntax",
                    token_start=alternative.get("token_start"),
                    token_end=alternative.get("token_end"),
                    source_object_id=clause.id,
                ))
            mode_hypothesis = InterpretationHypothesis(
                id=_stable_id("hypothesis", clause.id, "mode", clause.mode.value),
                scope_type="clause",
                scope_id=clause.id,
                hypothesis_type="mode",
                value=clause.mode.value,
            )
            hypotheses.append(mode_hypothesis)
            packets.extend([
                self._packet(
                    mode_hypothesis,
                    origin="clause_parser",
                    value=clause.mode.value,
                    support=0.86,
                    evidence_type="clause_structure",
                    group="syntax",
                    token_start=clause.token_start,
                    token_end=clause.token_end,
                    source_object_id=clause.id,
                ),
                self._packet(
                    mode_hypothesis,
                    origin="dialogue_act_parser",
                    value=clause.mode.value,
                    support=0.82,
                    evidence_type="overlapping_dialogue_act",
                    group="discourse",
                    token_start=clause.token_start,
                    token_end=clause.token_end,
                    source_object_id=clause.id,
                ),
            ])
            for predicate in clause.predicate_hypotheses:
                predicate_hypothesis = InterpretationHypothesis(
                    id=_stable_id(
                        "hypothesis",
                        clause.id,
                        "predicate",
                        predicate.get("token_index"),
                        predicate.get("lemma"),
                    ),
                    scope_type="clause",
                    scope_id=clause.id,
                    hypothesis_type=(
                        "embedded_predicate"
                        if predicate.get("embedded")
                        else "predicate"
                    ),
                    value={
                        "token_index": predicate.get("token_index"),
                        "lemma": predicate.get("lemma"),
                    },
                )
                hypotheses.append(predicate_hypothesis)
                packets.append(self._packet(
                    predicate_hypothesis,
                    origin="clause_parser",
                    value=predicate_hypothesis.value,
                    support=float(predicate.get("confidence", 0.5)),
                    evidence_type="predicate_center",
                    group="syntax",
                    token_start=predicate.get("token_index"),
                    token_end=predicate.get("token_index"),
                    source_object_id=clause.id,
                ))
                packets.append(self._packet(
                    predicate_hypothesis,
                    origin="morphology",
                    value=predicate_hypothesis.value,
                    support=0.72,
                    evidence_type="predicative_part_of_speech",
                    group="morphology",
                    token_start=predicate.get("token_index"),
                    token_end=predicate.get("token_index"),
                    source_object_id=clause.id,
                ))
            if clause.negation_scope:
                scope_hypothesis = InterpretationHypothesis(
                    id=_stable_id(
                        "hypothesis",
                        clause.id,
                        "negation_scope",
                        _canonical(clause.negation_scope),
                    ),
                    scope_type="clause",
                    scope_id=clause.id,
                    hypothesis_type="negation_scope",
                    value=clause.negation_scope,
                )
                hypotheses.append(scope_hypothesis)
                packets.extend([
                    self._packet(
                        scope_hypothesis,
                        origin="scope_parser",
                        value=clause.negation_scope,
                        support=float(clause.negation_scope.get("confidence", 0.7)),
                        evidence_type="operator_target_adjacency",
                        group="syntax",
                        token_start=clause.negation_scope.get("negation_token_index"),
                        token_end=clause.negation_scope.get("target_token_index"),
                        source_object_id=clause.id,
                    ),
                    self._packet(
                        scope_hypothesis,
                        origin="scope_parser",
                        value=clause.negation_scope,
                        support=0.68,
                        evidence_type="target_semantic_type",
                        group="semantics",
                        source_object_id=clause.id,
                    ),
                ])
            for participant in clause.participants:
                for role in participant.get("role_hypotheses", []):
                    role_hypothesis = InterpretationHypothesis(
                        id=_stable_id(
                            "hypothesis",
                            clause.id,
                            participant.get("token_start"),
                            "semantic_role",
                            role.get("role"),
                        ),
                        scope_type="participant",
                        scope_id=f"{clause.id}:{participant.get('token_start')}",
                        hypothesis_type="semantic_role",
                        value=role.get("role"),
                    )
                    hypotheses.append(role_hypothesis)
                    packets.append(self._packet(
                        role_hypothesis,
                        origin="clause_parser",
                        value=role.get("role"),
                        support=float(role.get("confidence", 0.5)),
                        evidence_type="grammatical_case",
                        group="morphology",
                        token_start=participant.get("token_start"),
                        token_end=participant.get("token_end"),
                        source_object_id=clause.id,
                    ))
        for token_index, candidates in (reference_candidates or {}).items():
            reference_scope_id = (
                f"{utterance_id}:reference:{token_index}"
                if utterance_id else str(token_index)
            )
            if not candidates:
                unresolved = InterpretationHypothesis(
                    id=_stable_id(
                        "hypothesis",
                        utterance_id,
                        "reference",
                        token_index,
                        "unresolved",
                    ),
                    scope_type="reference",
                    scope_id=reference_scope_id,
                    hypothesis_type="reference",
                    value=None,
                    unresolved_slots=["referent"],
                )
                hypotheses.append(unresolved)
                continue
            for candidate in candidates:
                hypothesis = InterpretationHypothesis(
                    id=_stable_id(
                        "hypothesis",
                        utterance_id,
                        "reference",
                        token_index,
                        candidate.get("id") or candidate.get("lemma"),
                    ),
                    scope_type="reference",
                    scope_id=reference_scope_id,
                    hypothesis_type="reference",
                    value=candidate,
                )
                hypotheses.append(hypothesis)
                evidence = candidate.get("evidence") or {}
                if isinstance(evidence, list):
                    evidence = {str(item): 0.5 for item in evidence}
                for group, support in evidence.items():
                    independent_group = (
                        group if group in {
                            "morphology",
                            "syntax",
                            "semantics",
                            "discourse",
                            "temporal_context",
                        } else "discourse"
                    )
                    packets.append(self._packet(
                        hypothesis,
                        origin="reference_resolver",
                        value=candidate,
                        support=float(support),
                        evidence_type=str(group),
                        group=independent_group,
                        token_start=token_index,
                        token_end=token_index,
                        source_object_id=str(candidate.get("id") or ""),
                    ))
        return hypotheses, packets

    @staticmethod
    def _apply_selected(
        analysis: LanguageAnalysis,
        hypotheses: Sequence[InterpretationHypothesis],
    ) -> None:
        selected = {
            (
                hypothesis.scope_type,
                hypothesis.scope_id,
                hypothesis.hypothesis_type,
            ): hypothesis
            for hypothesis in hypotheses
            if hypothesis.selected
        }
        for clause in analysis.clauses:
            predicate_winners = {
                (
                    hypothesis.value.get("token_index"),
                    hypothesis.value.get("lemma"),
                )
                for hypothesis in hypotheses
                if hypothesis.selected
                and hypothesis.scope_type == "clause"
                and hypothesis.scope_id == clause.id
                and hypothesis.hypothesis_type in {
                    "predicate",
                    "embedded_predicate",
                }
                and isinstance(hypothesis.value, Mapping)
            }
            for predicate in clause.predicate_hypotheses:
                predicate["selected"] = (
                    predicate.get("token_index"),
                    predicate.get("lemma"),
                ) in predicate_winners
            for participant in clause.participants:
                scope_id = (
                    f"{clause.id}:{participant.get('token_start')}"
                )
                winner = selected.get(
                    ("participant", scope_id, "semantic_role")
                )
                for role in participant.get("role_hypotheses", []):
                    role["selected"] = bool(
                        winner and role.get("role") == winner.value
                    )

    @staticmethod
    def _signature(
        analysis: LanguageAnalysis,
        hypotheses: Sequence[InterpretationHypothesis],
    ) -> str:
        selected = [
            (
                hypothesis.scope_type,
                hypothesis.scope_id,
                hypothesis.hypothesis_type,
                _canonical(hypothesis.value),
            )
            for hypothesis in hypotheses
            if hypothesis.selected
            and hypothesis.hypothesis_type in {
                "predicate",
                "clause_boundary",
                "semantic_role",
                "reference",
                "mode",
                "negation_scope",
            }
        ]
        payload = {
            "clause_boundaries": [
                (clause.token_start, clause.token_end) for clause in analysis.clauses
            ],
            "selected": sorted(selected),
        }
        return _canonical(payload)

    def _status(
        self,
        analysis: LanguageAnalysis,
        hypotheses: Sequence[InterpretationHypothesis],
    ) -> InterpretationStatus:
        unresolved = [
            hypothesis for hypothesis in hypotheses
            if hypothesis.unresolved_slots
        ]
        if unresolved:
            return InterpretationStatus.INCOMPLETE
        grouped: Dict[tuple[str, str, str], List[InterpretationHypothesis]] = {}
        for hypothesis in hypotheses:
            grouped.setdefault((
                hypothesis.scope_type,
                hypothesis.scope_id,
                hypothesis.hypothesis_type,
            ), []).append(hypothesis)
        critical_types = {
            "dialogue_act",
            "predicate",
            "clause_boundary",
            "semantic_role",
            "reference",
            "mode",
            "negation_scope",
        }
        for key, alternatives in grouped.items():
            if key[2] not in critical_types or len(alternatives) < 2:
                continue
            ranked = sorted(alternatives, key=lambda item: -item.support)
            if (
                ranked[0].support >= self.config.conflict_support
                and ranked[1].support >= self.config.conflict_support
                and any(
                    constraint.get("conflict")
                    or constraint.get("critical_violation")
                    for alternative in ranked[:2]
                    for constraint in alternative.constraints
                )
            ):
                return InterpretationStatus.CONFLICTED
            if (
                ranked[1].support >= self.config.ambiguity_support
                and ranked[0].support - ranked[1].support
                < self.config.ambiguity_margin
            ):
                return InterpretationStatus.AMBIGUOUS
        substantive_clauses = [
            clause for clause in analysis.clauses
            if clause.mode not in {ClauseMode.QUESTION, ClauseMode.REQUEST}
            or clause.predicate_hypotheses
        ]
        substantive_acts = [
            act for act in analysis.dialogue_acts
            if act.act_type not in {
                DialogueActType.GREETING,
                DialogueActType.SMALL_TALK,
                DialogueActType.DENIAL,
                DialogueActType.CONFIRMATION,
            }
        ]
        if substantive_acts and not substantive_clauses:
            return InterpretationStatus.INCOMPLETE
        return InterpretationStatus.STABLE

    def interpret(
        self,
        analysis: LanguageAnalysis,
        *,
        reference_candidates: Optional[
            Mapping[int, Sequence[Dict[str, Any]]]
        ] = None,
    ) -> LanguageAnalysis:
        hypotheses, packets = self._build(analysis, reference_candidates)
        cycles: List[Dict[str, Any]] = []
        previous_signature: Optional[str] = None
        stable_cycles = 0
        stop_reason = "MAX_CYCLES"
        for cycle in range(1, self.config.max_interpretation_cycles + 1):
            ranked = self.aggregator.rank(hypotheses, packets)
            signature = self._signature(analysis, ranked)
            if signature == previous_signature:
                stable_cycles += 1
            else:
                stable_cycles = 1
            for hypothesis in ranked:
                if hypothesis.selected:
                    hypothesis.stability_cycles = stable_cycles
                    hypothesis.status = (
                        HypothesisStatus.CONFIRMED
                        if hypothesis.support >= self.config.confirmation_support
                        and stable_cycles >= self.config.stable_cycles_required
                        else HypothesisStatus.PROVISIONAL
                    )
            cycles.append({
                "cycle": cycle,
                "signature": signature,
                "stable_cycles": stable_cycles,
                "selected_hypothesis_ids": [
                    hypothesis.id for hypothesis in ranked if hypothesis.selected
                ],
                "clause_boundaries": [
                    {
                        "clause_id": clause.id,
                        "token_start": clause.token_start,
                        "token_end": clause.token_end,
                    }
                    for clause in analysis.clauses
                ],
            })
            hypotheses = ranked
            if stable_cycles >= self.config.stable_cycles_required:
                stop_reason = "STABLE_INTERPRETATION"
                break
            previous_signature = signature
        status = self._status(analysis, hypotheses)
        if status != InterpretationStatus.STABLE:
            for hypothesis in hypotheses:
                if hypothesis.selected and hypothesis.status == HypothesisStatus.CONFIRMED:
                    hypothesis.status = HypothesisStatus.PROVISIONAL
        self._apply_selected(analysis, hypotheses)
        analysis.hypotheses = hypotheses
        analysis.evidence_packets = list({
            packet.dedupe_key: packet for packet in packets
        }.values())
        analysis.interpretation_status = status
        if analysis.utterance:
            analysis.utterance.interpretation_status = status
        analysis.interpretation_trace = {
            "config_version": self.config.version,
            "max_interpretation_cycles": self.config.max_interpretation_cycles,
            "stable_cycles_required": self.config.stable_cycles_required,
            "cycles": cycles,
            "cycles_completed": len(cycles),
            "stop_reason": stop_reason,
            "status": status.value,
            "independent_evidence_groups": sorted({
                packet.independent_group for packet in packets
            }),
        }
        return analysis
