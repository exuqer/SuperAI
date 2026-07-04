from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from semantic_ants.core.graph import SemanticGraph
from semantic_ants.core.models import AntRoute
from semantic_ants.generation.torch_dialogue import TorchDialogueNavigator
from semantic_ants.learning.checkpoint import Checkpoint


class Interpreter:
    """Преобразует маршруты в смысловое резюме и короткий ответ."""

    def __init__(self, navigator: TorchDialogueNavigator | None = None, model_dir: str | Path | None = None) -> None:
        self.navigator = navigator or TorchDialogueNavigator()
        self.model_dir = Path(model_dir) if model_dir is not None else None

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
    ) -> tuple[list[dict[str, Any]], str, str]:
        activated = self._rank_concepts(routes, graph, top_concepts)
        summary = self._summary(tokens, activated)
        response = (
            self._response(input_text, tokens, routes, activated, checkpoint, summary, chat_history)
            if generate_response
            else summary
        )
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
        routes: list[AntRoute],
        activated: list[dict[str, Any]],
        checkpoint: Checkpoint,
        summary: str,
        chat_history: list[dict[str, Any]] | None,
    ) -> str:
        prompt = self.navigator.build_prompt(
            input_text=input_text,
            tokens=tokens,
            activated_concepts=activated,
            routes=routes,
            checkpoint=checkpoint,
            chat_history=chat_history,
        )
        candidates = self.navigator.generate(
            prompt,
            checkpoint,
            model_dir=self.model_dir,
            fallback=summary,
            count=1,
        )
        return candidates[0] if candidates else summary


def _label_from_uri(uri: str) -> str:
    return uri.rstrip("/").split("/")[-1].replace("_", " ")
