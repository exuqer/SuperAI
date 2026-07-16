from __future__ import annotations

from typing import Any, Mapping


class HiveVisualTrace:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def add(self, event_type: str, payload: Mapping[str, Any], *, turn: int, tick: int) -> None:
        self.events.append(
            {"event_type": event_type, "turn": turn, "tick": tick, "payload": dict(payload)}
        )

    def to_dict(self) -> dict[str, Any]:
        return {"events": list(self.events)}
