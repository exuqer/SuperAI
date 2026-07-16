from __future__ import annotations

from collections import defaultdict

from .models import NectarPacket


class ObserverBee:
    bee_type = "observer"

    def allocate(self, packets: list[NectarPacket], budget: int) -> dict[str, int]:
        scores: dict[str, float] = defaultdict(float)
        for packet in packets:
            scores[packet.origin_space] += packet.utility * packet.confidence / max(1, packet.cost)
        total = sum(scores.values())
        if not total or budget <= 0:
            return {}
        allocations = {
            space: max(1, int(budget * score / total))
            for space, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        }
        overflow = sum(allocations.values()) - budget
        for space in reversed(list(allocations)):
            if overflow <= 0:
                break
            removable = min(overflow, max(0, allocations[space] - 1))
            allocations[space] -= removable
            overflow -= removable
        return allocations
