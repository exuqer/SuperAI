from __future__ import annotations

from typing import Any

from server.hive.hive_state import HiveState


class MultilevelTraceAnalytics:
    def query_trace(self, state: HiveState) -> dict[str, Any]:
        return dict(state.current_trace)

    def bee_trace(self, state: HiveState) -> dict[str, Any]:
        return {
            "tasks": state.active_tasks,
            "packets": state.nectar_packets,
            "ledger": state.current_trace.get("bees", []),
            "cost": sum(int(item.get("cost", 0)) for item in state.current_trace.get("bees", [])),
        }

    def memory_trace(self, state: HiveState) -> dict[str, Any]:
        return {
            "turn": state.memory.turn,
            "layers": state.memory.layer_counts(),
            "events": state.memory.events,
            "clusters": [item.to_dict() for item in state.memory.clusters.values()],
        }

    def factory_trace(self, state: HiveState) -> dict[str, Any]:
        return {"factories": state.factories, "vertical_transitions": state.vertical_transitions}

    def answer_trace(self, state: HiveState) -> dict[str, Any]:
        return {
            "answer": state.answer,
            "word_assembly": state.current_trace.get("word_assembly"),
            "rejected": state.current_trace.get("rejected", []),
        }

    def all(self, state: HiveState) -> dict[str, Any]:
        return {
            "query": self.query_trace(state),
            "bees": self.bee_trace(state),
            "memory": self.memory_trace(state),
            "factories": self.factory_trace(state),
            "answer": self.answer_trace(state),
        }
