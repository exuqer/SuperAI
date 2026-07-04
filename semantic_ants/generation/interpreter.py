from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from semantic_ants.core.graph import SemanticGraph
from semantic_ants.core.models import AntRoute
from semantic_ants.generation.torch_dialogue import TorchDialogueNavigator
from semantic_ants.generation.vector_interpreter import SemanticVectorInterpreter
from semantic_ants.learning.checkpoint import Checkpoint


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
    ) -> tuple[list[dict[str, Any]], str, str, dict[str, Any]]:
        activated = self._rank_concepts(routes, graph, checkpoint, top_concepts)
        vector_items = self._rank_concepts(routes, graph, checkpoint, max(top_concepts, 12))
        semantic_vector = self._semantic_vector(input_text, tokens, vector_items, routes, strength_vector)
        summary = self._summary(tokens, activated)
        response = (
            self._response(input_text, tokens, routes, activated, checkpoint, summary, chat_history, semantic_vector)
            if generate_response
            else summary
        )
        return activated, summary, response, semantic_vector

    def _rank_concepts(
        self,
        routes: list[AntRoute],
        graph: SemanticGraph,
        checkpoint: Checkpoint,
        top_concepts: int,
    ) -> list[dict[str, Any]]:
        scores: Counter[str] = Counter()
        sources: dict[str, set[str]] = {}
        for route in routes:
            route_score = max(route.total_score, 0.01)
            for index, concept in enumerate(route.concepts):
                scores[concept] += route_score / (index + 1)
                sources.setdefault(concept, set())
            for step in route.steps:
                sources.setdefault(step.end, set()).add(step.source)
        ranked = []
        for uri, score in scores.most_common(top_concepts):
            node = graph.nodes.get(uri)
            label = _label_for(uri, node, checkpoint)
            ranked.append(
                {
                    "uri": uri,
                    "label": label,
                    "language": node.language if node else "unknown",
                    "layer": node.layer if node else 1,
                    "score": round(float(score), 4),
                    "sources": sorted(sources.get(uri, set())),
                }
            )
        return ranked

    def _semantic_vector(
        self,
        input_text: str,
        tokens: list[str],
        items: list[dict[str, Any]],
        routes: list[AntRoute],
        strength_vector: tuple[int, ...],
    ) -> dict[str, Any]:
        layers: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            layer = str(item.get("layer", 1))
            layers.setdefault(layer, []).append(item)
        top_domain = next((item for item in items if int(item.get("layer", 1)) == 0), None)
        return {
            "version": 1,
            "input_text": input_text,
            "tokens": tokens,
            "strength_vector": list(strength_vector),
            "items": items,
            "layers": layers,
            "top_domain": top_domain,
            "routes": [
                {
                    "ant_id": route.ant_id,
                    "total_score": round(float(route.total_score), 4),
                    "concepts": route.concepts,
                }
                for route in routes[:8]
            ],
        }

    def _summary(self, tokens: list[str], activated: list[dict[str, Any]]) -> str:
        if not activated:
            return "Смысловые маршруты не найдены."
        labels = ", ".join(item["label"] for item in activated[:3])
        token_text = " ".join(tokens)
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
    ) -> str:
        response = self.vector_interpreter.interpret(semantic_vector, checkpoint, count=1)
        return response or summary


def _label_for(uri: str, node: Any, checkpoint: Checkpoint) -> str:
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
