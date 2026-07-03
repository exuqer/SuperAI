from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from semantic_ants.core.models import SemanticResult
from semantic_ants.learning.checkpoint import CheckpointStore


@dataclass
class TrainingReport:
    examples: int = 0
    epochs: int = 0
    reinforced_edges: int = 0
    suppressed_concepts: int = 0
    remembered_responses: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "examples": self.examples,
            "epochs": self.epochs,
            "reinforced_edges": self.reinforced_edges,
            "suppressed_concepts": self.suppressed_concepts,
            "remembered_responses": self.remembered_responses,
            "errors": self.errors,
        }


class Trainer:
    """Пакетное supervised-обучение по JSONL."""

    def __init__(self, engine: Any, store: CheckpointStore) -> None:
        self.engine = engine
        self.store = store

    def train_file(self, path: str | Path, epochs: int = 1) -> TrainingReport:
        examples = self._read_jsonl(path)
        report = TrainingReport(examples=len(examples), epochs=epochs)
        for _ in range(epochs):
            for index, example in enumerate(examples, start=1):
                try:
                    self.train_example(example, report)
                except (KeyError, TypeError, ValueError) as exc:
                    report.errors.append(f"{path}:{index}: {exc}")
        self.store.save(self.engine.checkpoint)
        return report

    def train_example(self, example: dict[str, Any], report: TrainingReport | None = None) -> None:
        text_value = example.get("text", example.get("stimulus"))
        if not text_value:
            raise ValueError("Пример требует text или stimulus")
        text = str(text_value)
        lang = str(example.get("lang", "auto"))
        result = self.engine.analyze(text, lang=lang, persist_result=False)
        checkpoint = self.engine.checkpoint
        target_concepts = [str(value) for value in example.get("target_concepts", [])]
        positive_edges = list(example.get("positive_edges", []))
        negative_concepts = [str(value) for value in example.get("negative_concepts", [])]
        rejected_answers = [str(value) for value in example.get("rejected_answers", [])]
        target_response = str(example.get("target_response") or example.get("accepted_answer") or "")
        reward = float(example.get("reward", 1.0))
        reinforced_any = False

        for route in result.routes:
            concepts = route.concepts
            for step in route.steps:
                if step.end in target_concepts or step.start in target_concepts:
                    checkpoint.reinforce_edge(step.start, step.relation, step.end, amount=0.35)
                    reinforced_any = True
                    if report:
                        report.reinforced_edges += 1
            for target in target_concepts:
                if target in concepts:
                    checkpoint.remember_response(target_concepts, target_response, amount=1.0)

        if target_concepts and not reinforced_any:
            for route in result.routes[:1]:
                for step in route.steps:
                    checkpoint.reinforce_edge(step.start, step.relation, step.end, amount=0.15)
                    reinforced_any = True
                    if report:
                        report.reinforced_edges += 1

        for edge_value in positive_edges:
            start, relation, end = _parse_positive_edge(edge_value)
            checkpoint.add_custom_edge(start, end, relation=relation, weight=1.0)
            checkpoint.reinforce_edge(start, relation, end, amount=0.5)
            if report:
                report.reinforced_edges += 1

        for concept in negative_concepts:
            checkpoint.suppress_concept(concept, amount=0.75)
            if report:
                report.suppressed_concepts += 1

        if target_response and target_concepts:
            checkpoint.remember_response(target_concepts, target_response, amount=1.0)
            checkpoint.remember_accepted_answer(
                stimulus=text,
                semantic_prompt=result.summary,
                concepts=target_concepts,
                answer=target_response,
                reward=reward,
            )
            if report:
                report.remembered_responses += 1

        for rejected in rejected_answers:
            checkpoint.remember_negative(
                stimulus=text,
                semantic_prompt=result.summary,
                concepts=target_concepts,
                answer=rejected,
                reason="supervised rejected answer",
            )

        checkpoint.examples_seen += 1

    def _read_jsonl(self, path: str | Path) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                items.append(json.loads(stripped))
        return items


class FeedbackTrainer:
    """Интерактивное подкрепление последнего или выбранного результата."""

    def __init__(self, engine: Any, store: CheckpointStore) -> None:
        self.engine = engine
        self.store = store

    def apply(
        self,
        result_id: str | None = None,
        score: int = 0,
        corrected_concepts: list[str] | None = None,
        corrected_response: str | None = None,
    ) -> dict[str, Any]:
        checkpoint = self.engine.checkpoint
        selected_id = result_id or checkpoint.last_result_id
        if not selected_id or selected_id not in checkpoint.results:
            raise ValueError("Не найден результат для feedback")
        result = checkpoint.results[selected_id]
        positive = score >= 4
        negative = score <= 2
        changed_edges = 0
        for route in result.get("routes", []):
            for step in route.get("steps", []):
                if positive:
                    checkpoint.reinforce_edge(step["start"], step["relation"], step["end"], amount=0.25)
                    changed_edges += 1
                elif negative:
                    checkpoint.penalize_edge(step["start"], step["relation"], step["end"], amount=0.25)
                    checkpoint.suppress_concept(step["end"], amount=0.25)
                    changed_edges += 1
        if corrected_concepts:
            for concept in corrected_concepts:
                checkpoint.suppressed_concepts.pop(concept, None)
            if corrected_response:
                checkpoint.remember_response(corrected_concepts, corrected_response, amount=2.0)
                checkpoint.remember_accepted_answer(
                    stimulus=str(result.get("input_text", "")),
                    semantic_prompt=str(result.get("summary", "")),
                    concepts=corrected_concepts,
                    answer=corrected_response,
                    reward=max(score / 5, 0.1),
                )
        elif positive and result.get("response"):
            concepts = [str(item.get("uri")) for item in result.get("activated_concepts", []) if item.get("uri")]
            checkpoint.remember_accepted_answer(
                stimulus=str(result.get("input_text", "")),
                semantic_prompt=str(result.get("summary", "")),
                concepts=concepts,
                answer=str(result["response"]),
                reward=max(score / 5, 0.1),
            )
        if negative and result.get("response"):
            concepts = [str(item.get("uri")) for item in result.get("activated_concepts", []) if item.get("uri")]
            checkpoint.remember_negative(
                stimulus=str(result.get("input_text", "")),
                semantic_prompt=str(result.get("summary", "")),
                concepts=concepts,
                answer=str(result["response"]),
                reason=f"human feedback score={score}",
            )
        self.store.save(checkpoint)
        return {
            "result_id": selected_id,
            "score": score,
            "positive": positive,
            "negative": negative,
            "changed_edges": changed_edges,
            "corrected_concepts": corrected_concepts or [],
        }


def _parse_positive_edge(value: Any) -> tuple[str, str, str]:
    if isinstance(value, dict):
        return (
            str(value["start"]),
            str(value.get("relation", "LearnedRelatedTo")),
            str(value["end"]),
        )
    if isinstance(value, list | tuple):
        if len(value) == 2:
            return str(value[0]), "LearnedRelatedTo", str(value[1])
        if len(value) == 3:
            return str(value[0]), str(value[1]), str(value[2])
    raise ValueError(f"Некорректное positive_edges значение: {value!r}")
