"""Public V2.7 hive facade over role-free event/query graphs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .graph_repository import GraphRepository
from .query_scene import QuerySceneService


class V2HiveService:
    def __init__(
        self,
        repository: Optional[GraphRepository] = None,
        config: Any = None,
    ) -> None:
        self.repository = repository or GraphRepository()
        self.query_scenes = QuerySceneService(self.repository)
        self.config = config

    def create(
        self,
        max_cells: int = 24,
        conversation_id: str = "",
    ) -> Dict[str, Any]:
        return self.query_scenes.create(max_cells, conversation_id)

    def get_hive(self, hive_id: str) -> Dict[str, Any]:
        return self.query_scenes.get(hive_id)

    def delete(self, hive_id: str) -> Dict[str, Any]:
        return self.query_scenes.delete(hive_id)

    def preview(self, hive_id: str, text: str) -> Dict[str, Any]:
        state = self.query_scenes.get(hive_id)
        previous_graph, previous_binding = (
            self.query_scenes.dialogue._previous(state)
        )
        parsed = self.query_scenes.dialogue.parse(
            text,
            previous_graph=previous_graph,
            previous_binding=previous_binding,
        )
        return {
            **parsed,
            "hive_id": hive_id,
            "mutated": False,
        }

    def query(
        self,
        hive_id: str,
        text: str,
        resolved_mode: Optional[str] = None,
        resonance_scope: str = "LOCAL_THEN_GLOBAL",
    ) -> Dict[str, Any]:
        result = self.query_scenes.query(
            hive_id,
            text,
            resolved_mode=resolved_mode,
            retrieval_scope=resonance_scope,
        )
        result["resolved_mode"] = (
            resolved_mode
            or (
                "FOLLOW_UP"
                if result["query_graph"].get("continuation_of")
                else "NEW_QUERY"
            )
        )
        result["retrieval_scope"] = resonance_scope
        return result

    def parse_query(self, text: str) -> Dict[str, Any]:
        return self.query_scenes.parse(text)

    def activate_query(
        self,
        hive_id: str,
        text: str,
        resolved_mode: str = "NEW_QUERY",
    ) -> Dict[str, Any]:
        return self.query_scenes.activate(hive_id, text, resolved_mode)

    def query_working_state(self, hive_id: str) -> Dict[str, Any]:
        return self.query_scenes.get(hive_id)

    def vibration_step(
        self,
        hive_id: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.query_scenes.step(hive_id, config)

    def vibration_run(
        self,
        hive_id: str,
        steps: int = 3,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = self.query_scenes.get(hive_id)
        return {
            "status": "FINISHED",
            "steps_completed": min(max(1, int(steps)), 32),
            "candidate_bindings": state.get("candidate_bindings", []),
            "answer": state.get("answer"),
            "hive": state,
            "config": dict(config or {}),
        }

    def vibration_stop(self, hive_id: str) -> Dict[str, Any]:
        return self.query_scenes.stop(hive_id)

    def dynamics_state(
        self,
        hive_id: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = self.query_scenes.get(hive_id)
        return {
            "status": "STRUCTURALLY_ADMITTED",
            "nodes": state.get("candidate_bindings", []),
            "config": dict(config or {}),
        }

    def dynamics_step(
        self,
        hive_id: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.dynamics_state(hive_id, config)

    def dynamics_history(self, hive_id: str) -> List[Dict[str, Any]]:
        state = self.query_scenes.get(hive_id)
        return [state.get("trace", {})] if state.get("trace") else []

    def dynamics_reset(self, hive_id: str) -> Dict[str, Any]:
        return {"status": "RESET", "hive_id": hive_id}

    def dynamics_node(self, hive_id: str, cell_id: str) -> Dict[str, Any]:
        state = self.query_scenes.get(hive_id)
        return next(
            (
                item for item in state.get("candidate_bindings", [])
                if item.get("binding_id") == cell_id
            ),
            {},
        )

    def dynamics_evictions(self, hive_id: str) -> List[Dict[str, Any]]:
        return self.query_scenes.get(hive_id).get("rejected_events", [])

    def events(self, hive_id: str) -> List[Dict[str, Any]]:
        with self.repository.transaction() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    """SELECT id,predicate_lemma,predicate_concept_id,
                              confidence,polarity,actuality
                       FROM graph_events ORDER BY created_at,id"""
                ).fetchall()
            ]

    def decisions(self, hive_id: str) -> List[Dict[str, Any]]:
        return self.query_scenes.get(hive_id).get("candidate_bindings", [])

    def matches(self, hive_id: str, cell_id: str) -> List[Dict[str, Any]]:
        node = self.dynamics_node(hive_id, cell_id)
        return [node] if node else []

    def snapshot(self, hive_id: str, **_: Any) -> Dict[str, Any]:
        return self.query_scenes.get(hive_id)

    def export(
        self,
        hive_id: str,
        mode: str = "current",
        run_id: Optional[str] = None,
        step: Optional[int] = None,
        detail: str = "full",
    ) -> Dict[str, Any]:
        return {
            "mode": mode,
            "run_id": run_id,
            "step": step,
            "detail": detail,
            "state": self.query_scenes.get(hive_id),
        }

    def analytics(
        self,
        hive_id: str,
        run_id: Optional[str] = None,
        compare_run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        state = self.query_scenes.get(hive_id)
        return {
            "hive_id": hive_id,
            "run_id": run_id,
            "compare_run_id": compare_run_id,
            "binding_count": len(state.get("candidate_bindings", [])),
            "rejected_event_count": len(state.get("rejected_events", [])),
            "answer_status": (state.get("answer") or {}).get("status"),
        }

    def forage(self, query: str, max_cells: int = 24) -> Dict[str, Any]:
        created = self.create(max_cells)
        return self.query(created["hive"]["id"], query)
