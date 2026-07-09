from __future__ import annotations

from typing import Any

from ..engine import SemanticEngine


def graph_snapshot(engine: SemanticEngine, *, query: str | None = None, limit: int = 120, result_id: str | None = None) -> dict[str, Any]:
    return engine.graph(query=query, limit=limit, result_id=result_id)


def concept_detail(engine: SemanticEngine, node_id: str) -> dict[str, Any]:
    return engine.node_detail(node_id)
