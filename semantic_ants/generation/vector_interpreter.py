from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    ) -> str:
        normalized = _normalize_vector(semantic_vector)
        fallback = self._fallback_sentence(normalized, checkpoint)
        prompt = self._prompt(normalized)
        candidates = self.navigator.generate(
            prompt,
            checkpoint,
            model_dir=self.model_dir,
            fallback=fallback,
            count=count,
        )
        return candidates[0] if candidates else fallback

    def _prompt(self, semantic_vector: dict[str, Any]) -> str:
        compact = {
            "input_text": semantic_vector.get("input_text", ""),
            "strength_vector": semantic_vector.get("strength_vector", []),
            "top_domain": semantic_vector.get("top_domain"),
            "items": semantic_vector.get("items", [])[:12],
        }
        return "\n".join(
            [
                "semantic_vector:",
                json.dumps(compact, ensure_ascii=False, sort_keys=True),
                "task: turn the semantic vector into one natural sentence",
                "assistant:",
            ]
        )

    def _fallback_sentence(self, semantic_vector: dict[str, Any], checkpoint: Checkpoint) -> str:
        items = [item for item in semantic_vector.get("items", []) if isinstance(item, dict)]
        top_domain = semantic_vector.get("top_domain")
        if not isinstance(top_domain, dict):
            top_domain = next((item for item in items if int(item.get("layer", 1)) == 0), None)
        domain_label = _label(top_domain, checkpoint) if isinstance(top_domain, dict) else ""
        concept_labels = [
            _label(item, checkpoint)
            for item in items
            if _label(item, checkpoint) and (not domain_label or _label(item, checkpoint) != domain_label)
        ][:4]
        if domain_label and concept_labels:
            return f"Смысловой вектор указывает на область «{domain_label}» и понятия: {', '.join(concept_labels)}."
        if domain_label:
            return f"Смысловой вектор указывает на область «{domain_label}»."
        if concept_labels:
            return f"Смысловой вектор связан с понятиями: {', '.join(concept_labels)}."
        return "Смысловой вектор пуст или недостаточно активирован."


def _normalize_vector(value: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(value, dict):
        items = value.get("items", [])
        return {**value, "items": items if isinstance(items, list) else []}
    if isinstance(value, list):
        return {"version": 1, "items": value}
    return {"version": 1, "items": []}


def _label(item: dict[str, Any] | None, checkpoint: Checkpoint) -> str:
    if not item:
        return ""
    label = str(item.get("label") or "")
    if label:
        return label
    uri = str(item.get("uri") or "")
    learned = _learned_label(uri, checkpoint)
    if learned:
        return learned
    return uri.rstrip("/").split("/")[-1].replace("_", " ")


def _learned_label(uri: str, checkpoint: Checkpoint) -> str:
    definitions = checkpoint.metadata.get("concept_definitions", {})
    if isinstance(definitions, dict):
        raw = definitions.get(uri)
        if isinstance(raw, dict) and raw.get("label"):
            return str(raw["label"])
    top_domains = checkpoint.metadata.get("top_domains", {})
    if isinstance(top_domains, dict):
        for raw in top_domains.values():
            if isinstance(raw, dict) and raw.get("uri") == uri and raw.get("label"):
                return str(raw["label"])
    labels = checkpoint.metadata.get("concept_labels", {})
    if isinstance(labels, dict) and labels.get(uri):
        return str(labels[uri])
    return ""
