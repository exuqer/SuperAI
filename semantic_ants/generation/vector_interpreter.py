from __future__ import annotations

from pathlib import Path
from typing import Any

from semantic_ants.core.normalization import detect_response_language
from semantic_ants.generation.sentences import select_vector_response
from semantic_ants.generation.torch_dialogue import TorchDialogueNavigator
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
        response_lang: str | None = None,
    ) -> str:
        normalized = _normalize_vector(semantic_vector)
        selected_lang = response_lang if response_lang in {"ru", "en"} else str(
            normalized.get("response_lang")
            or normalized.get("target_lang")
            or normalized.get("answer_lang")
            or ""
        )
        if selected_lang not in {"ru", "en"}:
            selected_lang = detect_response_language(
                str(normalized.get("input_text", "")),
                default=str(normalized.get("lang", "auto")),
            )
        if chat_history:
            normalized = {**normalized, "chat_history": chat_history[-8:]}
        response_vector = {
            **normalized,
            "lang": normalized.get("lang", "auto"),
            "response_lang": selected_lang or normalized.get("response_lang") or normalized.get("lang", "auto"),
        }
        selected = select_vector_response(
            response_vector,
            checkpoint,
            count=max(count, 4),
            navigator=self.navigator,
            model_dir=self.model_dir,
            creativity=float(response_vector.get("creativity", 0.35) or 0.35),
        )
        if not selected["candidates"] and response_vector.get("response_lang") != normalized.get("lang"):
            fallback_vector = {**normalized, "response_lang": normalized.get("lang", "auto")}
            selected = select_vector_response(
                fallback_vector,
                checkpoint,
                count=max(count, 4),
                navigator=self.navigator,
                model_dir=self.model_dir,
                creativity=float(fallback_vector.get("creativity", 0.35) or 0.35),
            )
        return str(selected.get("response") or next((candidate for candidate in selected.get("candidates", []) if candidate), ""))


def _normalize_vector(value: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(value, dict):
        items = value.get("items", [])
        return {**value, "items": items if isinstance(items, list) else []}
    if isinstance(value, list):
        return {"version": 1, "items": value}
    return {"version": 1, "items": []}


