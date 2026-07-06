from __future__ import annotations

from pathlib import Path
from typing import Any

from semantic_ants.generation.torch_dialogue import TorchDialogueNavigator
from semantic_ants.generation.sentences import build_vector_candidates
from semantic_ants.learning.checkpoint import Checkpoint


class SemanticVectorInterpreter:
    """Преобразует готовый смысловой вектор в человеческую фразу."""

    def __init__(self, navigator: TorchDialogueNavigator | None = None, model_dir: str | Path | None = None) -> None:
        self.navigator = navigator or TorchDialogueNavigator()
        self.model_dir = Path(model_dir) if model_dir is not None else None

    def interpret(
        self,
        semantic_vector: dict[str, Any] | list[dict[str, Any]],
        checkpoint: Checkpoint,
        count: int = 1,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> str:
        normalized = _normalize_vector(semantic_vector)
        if chat_history:
            normalized = {**normalized, "chat_history": chat_history[-8:]}
        candidates = build_vector_candidates(normalized, checkpoint, count=max(count, 4))
        return next((candidate for candidate in candidates if candidate), "")


def _normalize_vector(value: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(value, dict):
        items = value.get("items", [])
        return {**value, "items": items if isinstance(items, list) else []}
    if isinstance(value, list):
        return {"version": 1, "items": value}
    return {"version": 1, "items": []}


