from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from semantic_ants.generation.torch_dialogue import TorchDialogueNavigator
from semantic_ants.generation.sentences import build_vector_candidates, render_uri
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
        lang = _vector_lang(normalized)
        fallback_candidates = build_vector_candidates(normalized, checkpoint, count=max(count, 4))
        prompt = self._prompt(normalized)
        candidates = self.navigator.generate(
            prompt,
            checkpoint,
            model_dir=self.model_dir,
            fallback="",
            count=max(count, 4),
            lang=lang,
        )
        if not candidates:
            candidates = fallback_candidates
        selected = _select_candidate(candidates, normalized, lang)
        if selected:
            return selected
        return fallback_candidates[0] if fallback_candidates else "Смысл пока слишком разрежен для уверенного ответа."

    def _prompt(self, semantic_vector: dict[str, Any]) -> str:
        compact = {
            "lang": semantic_vector.get("lang", "auto"),
            "input_text": semantic_vector.get("input_text", ""),
            "strength_vector": semantic_vector.get("strength_vector", []),
            "top_domain": semantic_vector.get("top_domain"),
            "items": semantic_vector.get("items", [])[:12],
        }
        return "\n".join(
            [
                "semantic_vector:",
                json.dumps(compact, ensure_ascii=False, sort_keys=True),
                *_history_lines(semantic_vector.get("chat_history", [])),
                "task: turn the semantic vector into one natural sentence in the same language as the input",
                "assistant:",
            ]
        )


def _normalize_vector(value: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(value, dict):
        items = value.get("items", [])
        return {**value, "items": items if isinstance(items, list) else []}
    if isinstance(value, list):
        return {"version": 1, "items": value}
    return {"version": 1, "items": []}


def _vector_lang(semantic_vector: dict[str, Any]) -> str:
    lang = str(semantic_vector.get("lang", "auto"))
    if lang in {"ru", "en"}:
        return lang
    text = str(semantic_vector.get("input_text", ""))
    return "ru" if any(ch.lower() in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" for ch in text) else "en"


def _select_candidate(candidates: list[str], semantic_vector: dict[str, Any], lang: str) -> str:
    values = [candidate for candidate in candidates if candidate and _language_matches(candidate, lang)]
    if not values:
        values = [candidate for candidate in candidates if candidate]
    if not values:
        return ""
    seed = hashlib.sha256(
        "|".join(
            [
                str(semantic_vector.get("input_text", "")),
                lang,
                str(semantic_vector.get("top_domain", {})),
                ",".join(str(item.get("uri", "")) for item in semantic_vector.get("items", [])[:6] if isinstance(item, dict)),
                _history_fingerprint(semantic_vector.get("chat_history", [])),
            ]
        ).encode("utf-8")
    ).hexdigest()
    return values[int(seed[:2], 16) % len(values)]


def _history_lines(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    lines = ["chat_history:"]
    for turn in value[-8:]:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "user"))
        text = " ".join(str(turn.get("text", "")).split())
        if text:
            lines.append(f"{role}: {text}")
    return lines if len(lines) > 1 else []


def _history_fingerprint(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for turn in value[-4:]:
        if isinstance(turn, dict):
            parts.append(f"{turn.get('role', '')}:{turn.get('text', '')}")
    return "|".join(parts)


def _language_matches(text: str, lang: str) -> bool:
    if lang == "ru":
        return any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in text)
    if lang == "en":
        return not any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in text)
    return True


def _label(item: dict[str, Any] | None, checkpoint: Checkpoint) -> str:
    if not item:
        return ""
    uri = str(item.get("uri") or "")
    label = render_uri(uri, checkpoint, str(item.get("language") or "auto"))
    if label:
        return label
    raw_label = str(item.get("label") or "")
    if raw_label:
        return raw_label
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
