from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from semantic_ants.core.graph import SemanticGraph
from semantic_ants.core.models import AntRoute
from semantic_ants.core.normalization import detect_language, detect_response_language, tokenize
from semantic_ants.generation.torch_dialogue import TorchDialogueNavigator
from semantic_ants.generation.sentences import render_uri, select_vector_response
from semantic_ants.generation.vector_interpreter import SemanticVectorInterpreter
from semantic_ants.learning.checkpoint import Checkpoint
from semantic_ants.understanding import understand_text


class Interpreter:
    """Преобразует маршруты в смысловое резюме и короткий ответ."""

    def __init__(self, navigator: TorchDialogueNavigator | None = None, model_dir: str | Path | None = None) -> None:
        self.navigator = navigator or TorchDialogueNavigator()
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.vector_interpreter = SemanticVectorInterpreter(navigator=self.navigator, model_dir=self.model_dir)

    def interpret(
        self,
        input_text: str,
        tokens: list[str],
        routes: list[AntRoute],
        graph: SemanticGraph,
        checkpoint: Checkpoint,
        top_concepts: int = 5,
        chat_history: list[dict[str, Any]] | None = None,
        generate_response: bool = True,
        strength_vector: tuple[int, ...] = (),
        lang: str | None = None,
    ) -> tuple[list[dict[str, Any]], str, str, dict[str, Any]]:
        selected_lang = lang if lang in {"ru", "en"} else detect_language(input_text)
        response_lang = detect_response_language(input_text, default=selected_lang) or selected_lang
        token_focus = self._token_focus(input_text, checkpoint, selected_lang, chat_history)
        activated = self._rank_concepts(routes, graph, checkpoint, top_concepts, response_lang, token_focus)
        vector_items = self._rank_concepts(routes, graph, checkpoint, max(top_concepts, 12), response_lang, token_focus)
        semantic_vector = self._semantic_vector(
            input_text,
            tokens,
            vector_items,
            routes,
            strength_vector,
            selected_lang,
            response_lang,
            token_focus,
            chat_history,
        )
        summary = self._summary(tokens, activated, response_lang)
        response_data = self._response(
            input_text,
            tokens,
            routes,
            activated,
            checkpoint,
            summary,
            chat_history,
            semantic_vector,
            response_lang,
        )
        response = response_data["response"] if generate_response else summary
        semantic_vector = {**semantic_vector, "response_candidates": response_data["candidates"], "response_source": response_data["source"]}
        return activated, summary, response, semantic_vector

    def _rank_concepts(
        self,
        routes: list[AntRoute],
        graph: SemanticGraph,
        checkpoint: Checkpoint,
        top_concepts: int,
        lang: str,
        token_focus: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        scores: Counter[str] = Counter()
        sources: dict[str, set[str]] = {}
        has_content_focus = bool(token_focus)
        default_focus = 0.4 if has_content_focus else 1.0
        for route in routes:
            route_score = max(route.total_score, 0.01)
            for index, concept in enumerate(route.concepts):
                focus = token_focus.get(concept, default_focus) if token_focus else 1.0
                scores[concept] += route_score / (index + 1) * focus
                sources.setdefault(concept, set())
            for step in route.steps:
                sources.setdefault(step.end, set()).add(step.source)
        ranked = []
        for uri, score in scores.most_common(top_concepts):
            node = graph.nodes.get(uri)
            label = _label_for(uri, node, checkpoint, lang)
            layers = list(getattr(node, "layers", []) or ([node.layer] if node else [1]))
            active_layers = list(getattr(node, "active_layers", []) or layers)
            ranked.append(
                {
                    "uri": uri,
                    "label": label,
                    "language": node.language if node else "unknown",
                    "layer": layers[0] if layers else 1,
                    "layers": layers,
                    "active_layers": active_layers,
                    "score": round(float(score), 4),
                    "sources": sorted(sources.get(uri, set())),
                }
            )
        return ranked

    def _token_focus(self, input_text: str, checkpoint: Checkpoint, lang: str, chat_history: list[dict[str, Any]] | None) -> dict[str, float]:
        understanding = understand_text(input_text, lang=lang, checkpoint=checkpoint)
        focus: dict[str, float] = {}
        for token in understanding.tokens:
            if token.is_stop_word or not token.concept_uri:
                continue
            focus[token.concept_uri] = max(focus.get(token.concept_uri, 0.0), _token_focus_weight(token))
        if chat_history and _is_follow_up_text(input_text):
            decay = 0.75
            for turn in reversed(chat_history[-6:]):
                if str(turn.get("role", "")) != "user":
                    continue
                for concept in turn.get("concepts", []) or []:
                    concept_uri = str(concept)
                    if not concept_uri:
                        continue
                    focus[concept_uri] = max(focus.get(concept_uri, 0.0), decay)
                decay *= 0.7
        return focus

    def _semantic_vector(
        self,
        input_text: str,
        tokens: list[str],
        items: list[dict[str, Any]],
        routes: list[AntRoute],
        strength_vector: tuple[int, ...],
        source_lang: str,
        response_lang: str,
        context_focus: dict[str, float],
        chat_history: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        layers: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            item_layers = item.get("active_layers") or item.get("layers") or [item.get("layer", 1)]
            if not isinstance(item_layers, list):
                item_layers = [item_layers]
            for layer in item_layers:
                layers.setdefault(str(layer), []).append(item)
        top_domain = next(
            (
                item
                for item in items
                if 0 in _item_layers(item)
            ),
            None,
        )
        return {
            "version": 1,
            "lang": source_lang,
            "source_lang": source_lang,
            "response_lang": response_lang,
            "context_focus": context_focus,
            "input_text": input_text,
            "tokens": tokens,
            "strength_vector": list(strength_vector),
            "items": items,
            "layers": layers,
            "top_domain": top_domain,
            "chat_history": chat_history or [],
            "routes": [
                {
                    "ant_id": route.ant_id,
                    "total_score": round(float(route.total_score), 4),
                    "concepts": route.concepts,
                }
                for route in routes[:8]
            ],
        }

    def _summary(self, tokens: list[str], activated: list[dict[str, Any]], lang: str) -> str:
        if not activated:
            return "No semantic routes found." if lang == "en" else "Смысловые маршруты не найдены."
        labels = ", ".join(item["label"] for item in activated[:3])
        token_text = " ".join(tokens)
        if lang == "en":
            return f'The phrase "{token_text}" is connected with concepts: {labels}.'
        return f"Фраза «{token_text}» связана с концептами: {labels}."

    def _response(
        self,
        input_text: str,
        tokens: list[str],
        routes: list[AntRoute],
        activated: list[dict[str, Any]],
        checkpoint: Checkpoint,
        summary: str,
        chat_history: list[dict[str, Any]] | None,
        semantic_vector: dict[str, Any],
        response_lang: str | None,
    ) -> dict[str, Any]:
        selected = select_vector_response(
            semantic_vector,
            checkpoint,
            count=3,
            navigator=self.navigator,
            model_dir=self.model_dir,
            creativity=float(semantic_vector.get("creativity", 0.35) or 0.35),
        )
        return {
            "response": selected["response"],
            "candidates": selected["candidates"],
            "source": selected["source"],
        }


def _item_layers(item: dict[str, Any]) -> list[int]:
    layers = item.get("active_layers") or item.get("layers") or [item.get("layer", 1)]
    if not isinstance(layers, list):
        layers = [layers]
    values: list[int] = []
    for layer in layers:
        try:
            numeric = int(layer)
        except (TypeError, ValueError):
            continue
        if numeric not in values:
            values.append(numeric)
    return values


def _label_for(uri: str, node: Any, checkpoint: Checkpoint, lang: str) -> str:
    localized = render_uri(uri, checkpoint, lang)
    if localized:
        return localized
    learned = _learned_label(uri, checkpoint)
    if learned:
        return learned
    if node is not None and getattr(node, "label", None):
        return str(node.label)
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


def _token_focus_weight(token: Any) -> float:
    match str(getattr(token, "match_status", "")):
        case "found_as_alias" | "found_as_lemma" | "found_as_raw":
            return 2.5
        case "partial_root_match" | "edit_distance_match":
            return 2.0
        case _:
            return 1.8


def _is_follow_up_text(value: str) -> bool:
    tokens = tokenize(value)
    if not tokens:
        return False
    normalized = " ".join(tokens).casefold()
    if normalized in {
        "это",
        "что это",
        "а это",
        "подробнее",
        "расскажи подробнее",
        "ещё",
        "еще",
        "продолжай",
        "почему",
        "как именно",
        "what about it",
        "tell me more",
        "more",
        "continue",
        "why",
    }:
        return True
    follow_up_terms = {"это", "он", "она", "они", "подробнее", "ещё", "еще", "more", "it", "this", "that"}
    return len(tokens) <= 3 and all(token.casefold() in follow_up_terms for token in tokens)
