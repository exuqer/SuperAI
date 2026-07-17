"""Compatibility without collapsing distinct semantic roles."""

from __future__ import annotations

from typing import Dict


class RoleCompatibilityGraph:
    _GRAPH: Dict[str, Dict[str, float]] = {
        "agent": {"agent": 1.0, "theme": .45, "cause": .42},
        "cause": {"cause": 1.0, "agent": .42, "theme": .30},
        "patient": {"patient": 1.0, "theme": .72, "object": .60},
        "theme": {
            "theme": 1.0, "patient": .65, "object": .60, "agent": .42,
        },
        "object": {"object": 1.0, "patient": .60, "theme": .60},
        "recipient": {"recipient": 1.0, "experiencer": .45, "object": .25},
        "experiencer": {"experiencer": 1.0, "recipient": .45, "theme": .30},
        "location": {"location": 1.0, "destination": .52, "source": .20},
        "destination": {"destination": 1.0, "location": .48},
        "source": {"source": 1.0, "location": .25},
        "instrument": {"instrument": 1.0, "material": .52, "object": .20},
        "material": {"material": 1.0, "instrument": .52, "theme": .25},
    }

    def score(self, requested: str, observed: str) -> float:
        if not requested or not observed:
            return 0.0
        if requested == observed:
            return 1.0
        return self._GRAPH.get(requested, {}).get(observed, 0.0)
