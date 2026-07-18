"""Independent evidence aggregation for interpretation hypotheses."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite, prod
from typing import Dict, Iterable, List, Sequence

from .models import EvidencePacket, InterpretationHypothesis


INDEPENDENT_GROUPS = {
    "morphology",
    "agreement",
    "syntax",
    "valency",
    "semantics",
    "discourse",
    "temporal_context",
    "source",
    "manual_validation",
}


def clamp(value: float) -> float:
    number = float(value)
    if not isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


class EvidenceAggregator:
    """Aggregate dependent packets inside a group before combining groups."""

    def aggregate(
        self,
        hypothesis: InterpretationHypothesis,
        packets: Sequence[EvidencePacket],
    ) -> InterpretationHypothesis:
        unique: Dict[tuple[object, ...], EvidencePacket] = {}
        for packet in packets:
            if packet.target_hypothesis_id != hypothesis.id:
                continue
            if packet.independent_group not in INDEPENDENT_GROUPS:
                raise ValueError(
                    f"unknown independent evidence group: "
                    f"{packet.independent_group}"
                )
            current = unique.get(packet.dedupe_key)
            if current is None or (
                packet.support - packet.penalty,
                packet.support,
                -packet.penalty,
                packet.id,
            ) > (
                current.support - current.penalty,
                current.support,
                -current.penalty,
                current.id,
            ):
                unique[packet.dedupe_key] = packet
        relevant = list(unique.values())
        by_group: Dict[str, List[EvidencePacket]] = defaultdict(list)
        for packet in relevant:
            by_group[packet.independent_group].append(packet)
        group_support: Dict[str, float] = {}
        total_penalty = 0.0
        for group, items in by_group.items():
            positive = 1.0 - prod(
                1.0 - clamp(item.support)
                for item in items
            )
            penalty = 1.0 - prod(
                1.0 - clamp(item.penalty)
                for item in items
            )
            group_support[group] = round(clamp(positive - penalty), 6)
            total_penalty += penalty
        if group_support:
            independent_support = 1.0 - prod(
                1.0 - value for value in group_support.values()
            )
            diversity = min(1.0, len(group_support) / 3.0)
            hypothesis.support = round(
                clamp(0.82 * independent_support + 0.18 * diversity - 0.08 * total_penalty),
                6,
            )
        else:
            hypothesis.support = 0.0
        hypothesis.support_by_group = group_support
        return hypothesis

    def rank(
        self,
        hypotheses: Sequence[InterpretationHypothesis],
        packets: Sequence[EvidencePacket],
    ) -> List[InterpretationHypothesis]:
        grouped: Dict[tuple[str, str, str], List[InterpretationHypothesis]] = defaultdict(list)
        for hypothesis in hypotheses:
            self.aggregate(hypothesis, packets)
            grouped[(
                hypothesis.scope_type,
                hypothesis.scope_id,
                hypothesis.hypothesis_type,
            )].append(hypothesis)
        for alternatives in grouped.values():
            alternatives.sort(key=lambda item: (-item.support, str(item.value), item.id))
            leader = alternatives[0]
            runner_support = alternatives[1].support if len(alternatives) > 1 else 0.0
            leader.leader_margin = round(leader.support - runner_support, 6)
            leader.selected = True
            for alternative in alternatives[1:]:
                alternative.leader_margin = round(
                    alternative.support - leader.support,
                    6,
                )
                alternative.selected = False
        return sorted(
            hypotheses,
            key=lambda item: (
                item.scope_type,
                item.scope_id,
                item.hypothesis_type,
                -item.support,
                item.id,
            ),
        )

    @staticmethod
    def independent_source_count(packets: Iterable[EvidencePacket]) -> int:
        source_keys = {
            (
                packet.independent_group,
                packet.source_object_id or packet.origin,
            )
            for packet in packets
            if packet.support > packet.penalty
        }
        return len(source_keys)
