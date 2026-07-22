"""Public query service backed exclusively by V3.0 QueryGraph."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .graph_repository import GraphRepository
from .graph_service import GraphDialogueService
from .russian_morphology import RussianMorphology


class QuerySceneService:
    """Application boundary for QueryGraph parsing and dialogue."""

    def __init__(self, repository: Optional[GraphRepository] = None) -> None:
        self.repository = repository or GraphRepository()
        self.dialogue = GraphDialogueService(
            self.repository,
            RussianMorphology(),
        )

    def create(
        self,
        max_cells: int = 24,
        conversation_id: str = "",
    ) -> Dict[str, Any]:
        return self.dialogue.create(max_cells, conversation_id)

    def parse(self, text: str, **_: Any) -> Dict[str, Any]:
        return self.dialogue.parse(text)

    def query(
        self,
        hive_id: str,
        text: str,
        *,
        resolved_mode: Optional[str] = None,
        retrieval_scope: str = "LOCAL_THEN_GLOBAL",
        **_: Any,
    ) -> Dict[str, Any]:
        return self.dialogue.query(
            hive_id,
            text,
            resolved_mode=resolved_mode,
            retrieval_scope=retrieval_scope,
        )

    def activate(
        self,
        hive_id: str,
        text: str,
        resolved_mode: str = "NEW_QUERY",
    ) -> Dict[str, Any]:
        result = self.dialogue.query(
            hive_id,
            text,
            resolved_mode=resolved_mode,
        )
        result["resolved_mode"] = resolved_mode
        return result

    def get(self, hive_id: str) -> Dict[str, Any]:
        return self.dialogue.get(hive_id)

    def delete(self, hive_id: str) -> Dict[str, Any]:
        return self.dialogue.delete(hive_id)

    def query_working_state(self, hive_id: str) -> Dict[str, Any]:
        return self.get(hive_id)

    def resolve_mode(
        self,
        hive_id: str,
        text: str,
        resolved_mode: Optional[str] = None,
    ) -> str:
        if resolved_mode:
            return resolved_mode
        state = self.get(hive_id)
        return "FOLLOW_UP" if state.get("query_graph") else "NEW_QUERY"

    def step(
        self,
        hive_id: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = self.get(hive_id)
        return {
            "status": "STABLE",
            "config": dict(config or {}),
            "hive": state,
            "candidate_bindings": state.get("candidate_bindings", []),
        }

    def stop(self, hive_id: str) -> Dict[str, Any]:
        return {"status": "STOPPED", "hive": self.get(hive_id)}
