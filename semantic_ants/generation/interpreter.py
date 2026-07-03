from __future__ import annotations

from collections import Counter
from typing import Any

from semantic_ants.core.graph import SemanticGraph
from semantic_ants.core.models import AntRoute
from semantic_ants.generation.dialogue import DialogueResponder
from semantic_ants.learning.checkpoint import Checkpoint, concept_set_key


class Interpreter:
    """Преобразует маршруты в смысловое резюме и короткий ответ."""

    def __init__(self) -> None:
        self.dialogue = DialogueResponder()

    def interpret(
        self,
        input_text: str,
        tokens: list[str],
        routes: list[AntRoute],
        graph: SemanticGraph,
        checkpoint: Checkpoint,
        top_concepts: int = 5,
    ) -> tuple[list[dict[str, Any]], str, str]:
        activated = self._rank_concepts(routes, graph, top_concepts)
        summary = self._summary(tokens, activated)
        response = self._response(input_text, tokens, activated, checkpoint)
        return activated, summary, response

    def _rank_concepts(
        self,
        routes: list[AntRoute],
        graph: SemanticGraph,
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
            ranked.append(
                {
                    "uri": uri,
                    "label": node.label if node else _label_from_uri(uri),
                    "language": node.language if node else "unknown",
                    "score": round(float(score), 4),
                    "sources": sorted(sources.get(uri, set())),
                }
            )
        return ranked

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
        activated: list[dict[str, Any]],
        checkpoint: Checkpoint,
    ) -> str:
        concepts = [item["uri"] for item in activated]
        dialogue_response = self.dialogue.response_for(input_text, tokens, activated, checkpoint)
        if dialogue_response:
            return dialogue_response
        remembered = self._find_remembered_response(concepts, checkpoint)
        if remembered:
            return remembered
        if not activated:
            return "Пока не хватает связей в графе, чтобы уверенно выделить смысл."
        first = activated[0]["label"]
        if len(activated) == 1:
            return f"Главный смысл запроса сейчас выглядит как «{first}»."
        rest = ", ".join(item["label"] for item in activated[1:3])
        return f"Я вижу основной смысл «{first}» и связанные оттенки: {rest}."

    def _find_remembered_response(self, concepts: list[str], checkpoint: Checkpoint) -> str | None:
        if not concepts:
            return None
        concept_set = set(concepts)
        best_response: str | None = None
        best_score = 0.0
        for key, value in checkpoint.response_memory.items():
            memory_concepts = set(key.split("|"))
            overlap = len(concept_set & memory_concepts)
            if not overlap:
                continue
            score = overlap * float(value.get("weight", 0.0))
            if score > best_score:
                best_score = score
                best_response = str(value.get("response", ""))
        return best_response or None


def _label_from_uri(uri: str) -> str:
    return uri.rstrip("/").split("/")[-1].replace("_", " ")
