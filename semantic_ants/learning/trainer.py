from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from semantic_ants.core.models import SemanticResult
from semantic_ants.core.normalization import detect_response_language
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
        text_value = example.get("text") or example.get("stimulus") or example.get("question")
        if not text_value:
            raise ValueError("Пример требует text, stimulus или question")
        text = str(text_value)
        lang = str(example.get("lang", "auto"))
        answer_lang_value = str(
            example.get("answer_lang")
            or example.get("response_lang")
            or example.get("target_lang")
            or lang
        )
        answer_lang = answer_lang_value if answer_lang_value in {"ru", "en"} else (lang if lang in {"ru", "en"} else None)
        source_lang = lang if lang in {"ru", "en"} else None
        result = self.engine.analyze(
            text,
            lang=lang,
            persist_result=False,
            strength_vector=_parse_strength_vector(example.get("strength_vector")) or None,
        )
        checkpoint = self.engine.checkpoint
        target_concepts = [str(value) for value in example.get("target_concepts", [])]
        for concepts in _parse_layer_targets(example.get("layer_targets")).values():
            target_concepts.extend(concepts)
        target_concepts = list(dict.fromkeys(target_concepts))
        positive_edges = _parse_positive_edges(example.get("positive_edges"))
        negative_concepts = [str(value) for value in example.get("negative_concepts", [])]
        rejected_answers = [str(value) for value in example.get("rejected_answers", [])]
        target_response = str(
            example.get("target_response")
            or example.get("accepted_answer")
            or example.get("expected_answer")
            or ""
        )
        reward = float(example.get("reward", 1.0))
        reinforced_any = False

        for concept, label in _parse_concept_labels(example.get("concept_labels")).items():
            checkpoint.remember_concept_label(checkpoint.canonical_uri(concept), label)

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
                    checkpoint.remember_response(
                        target_concepts,
                        target_response,
                        amount=1.0,
                        lang=answer_lang,
                        source_lang=source_lang,
                    )

        if target_concepts and not reinforced_any:
            for route in result.routes[:1]:
                for step in route.steps:
                    checkpoint.reinforce_edge(step.start, step.relation, step.end, amount=0.15)
                    reinforced_any = True
                    if report:
                        report.reinforced_edges += 1

        for edge_value in positive_edges:
            if isinstance(edge_value, dict):
                start = str(edge_value["start"])
                relation = str(edge_value.get("relation", "LearnedRelatedTo"))
                end = str(edge_value["end"])
                weight = float(edge_value.get("weight", 1.0))
                layer = int(edge_value.get("layer", 1))
                distance = float(edge_value.get("distance", 1.0))
                edge_type = str(edge_value.get("edge_type", "semantic"))
                metadata = dict(edge_value.get("metadata", {})) if isinstance(edge_value.get("metadata"), dict) else {}
                checkpoint.add_custom_edge(
                    checkpoint.canonical_uri(start),
                    checkpoint.canonical_uri(end),
                    relation=relation,
                    weight=weight,
                    layer=layer,
                    distance=distance,
                    edge_type=edge_type,
                    metadata=metadata,
                )
                checkpoint.reinforce_edge(start, relation, end, amount=max(weight, 0.5))
                if report:
                    report.reinforced_edges += 1
                continue
            start, relation, end = edge_value
            checkpoint.add_custom_edge(checkpoint.canonical_uri(start), checkpoint.canonical_uri(end), relation=relation, weight=1.0)
            checkpoint.reinforce_edge(start, relation, end, amount=0.5)
            if report:
                report.reinforced_edges += 1

        layer_targets = _parse_layer_targets(example.get("layer_targets"))
        route_starts = list(dict.fromkeys(route.start for route in result.routes))
        for layer_key, concepts in layer_targets.items():
            layer = int(layer_key)
            relation = "InTopDomain" if layer == 0 else "LayerTarget"
            edge_type = "domain" if layer == 0 else "semantic"
            for start in route_starts:
                for concept in concepts:
                    if start == concept:
                        continue
                    checkpoint.add_custom_edge(
                        start,
                        concept,
                        relation=relation,
                        weight=1.0,
                        layer=layer,
                        distance=1.0,
                        edge_type=edge_type,
                        metadata={"layer_target": layer_key},
                    )
                    checkpoint.reinforce_edge(start, relation, concept, amount=0.4)
                    if report:
                        report.reinforced_edges += 1

        for concept in negative_concepts:
            checkpoint.suppress_concept(concept, amount=0.75)
            if report:
                report.suppressed_concepts += 1

        if target_response and target_concepts:
            checkpoint.remember_response(
                target_concepts,
                target_response,
                amount=1.0,
                lang=answer_lang,
                source_lang=source_lang,
            )
            checkpoint.remember_accepted_answer(
                stimulus=text,
                semantic_prompt=result.summary,
                concepts=target_concepts,
                answer=target_response,
                reward=reward,
                lang=answer_lang,
                source_lang=source_lang,
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
                lang=answer_lang,
                source_lang=source_lang,
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
        trained_dialogues = 0
        rejected_dialogues = 0
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
                answer_lang = detect_response_language(
                    str(result.get("input_text", "")),
                    default=str(result.get("lang", "auto")),
                )
                self.engine.speech.train_pair(
                    stimulus=str(result.get("input_text", "")),
                    semantic_prompt=str(result.get("summary", "")),
                    concepts=corrected_concepts,
                    accepted_answer=corrected_response,
                    checkpoint=checkpoint,
                    reward=max(score / 5, 0.1),
                    model_dir=self.engine.model_dir,
                    answer_lang=answer_lang,
                )
                trained_dialogues += 1
        elif positive and result.get("response"):
            concepts = [str(item.get("uri")) for item in result.get("activated_concepts", []) if item.get("uri")]
            answer_lang = detect_response_language(
                str(result.get("input_text", "")),
                default=str(result.get("lang", "auto")),
            )
            self.engine.speech.train_pair(
                stimulus=str(result.get("input_text", "")),
                semantic_prompt=str(result.get("summary", "")),
                concepts=concepts,
                accepted_answer=str(result["response"]),
                checkpoint=checkpoint,
                reward=max(score / 5, 0.1),
                model_dir=self.engine.model_dir,
                answer_lang=answer_lang,
            )
            trained_dialogues += 1
        if negative and result.get("response"):
            concepts = [str(item.get("uri")) for item in result.get("activated_concepts", []) if item.get("uri")]
            answer_lang = detect_response_language(
                str(result.get("input_text", "")),
                default=str(result.get("lang", "auto")),
            )
            self.engine.speech.train_negative(
                stimulus=str(result.get("input_text", "")),
                semantic_prompt=str(result.get("summary", "")),
                concepts=concepts,
                rejected_answer=str(result["response"]),
                checkpoint=checkpoint,
                reason=f"human feedback score={score}",
                answer_lang=answer_lang,
            )
            rejected_dialogues += 1
        self.store.save(checkpoint)
        return {
            "result_id": selected_id,
            "score": score,
            "positive": positive,
            "negative": negative,
            "changed_edges": changed_edges,
            "trained_dialogues": trained_dialogues,
            "rejected_dialogues": rejected_dialogues,
            "corrected_concepts": corrected_concepts or [],
        }


def _parse_positive_edge(value: Any) -> tuple[str, str, str] | dict[str, Any]:
    if isinstance(value, dict):
        return _parse_positive_edge_dict(value)
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            return str(value[0]), "LearnedRelatedTo", str(value[1])
        if len(value) == 3:
            return str(value[0]), str(value[1]), str(value[2])
    raise ValueError(f"Некорректное positive_edges значение: {value!r}")


def _parse_positive_edges(value: Any) -> list[tuple[str, str, str] | dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict) and {"start", "end"} <= set(value):
        return [_parse_positive_edge_dict(value)]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Некорректное positive_edges значение: {value!r}")
    return [_parse_positive_edge(edge) for edge in value]


def _parse_positive_edge_dict(value: dict[str, Any]) -> dict[str, Any]:
    if "start" not in value or "end" not in value:
        raise ValueError(f"Некорректное positive_edges значение: {value!r}")
    edge: dict[str, Any] = {
        "start": str(value["start"]),
        "relation": str(value.get("relation", "LearnedRelatedTo")),
        "end": str(value["end"]),
    }
    if "weight" in value and value["weight"] is not None:
        edge["weight"] = float(value["weight"])
    if "layer" in value and value["layer"] is not None:
        edge["layer"] = int(value["layer"])
    if "distance" in value and value["distance"] is not None:
        edge["distance"] = float(value["distance"])
    if "edge_type" in value and value["edge_type"] is not None:
        edge["edge_type"] = str(value["edge_type"])
    metadata = value.get("metadata", {})
    if metadata is not None:
        edge["metadata"] = dict(metadata) if isinstance(metadata, dict) else dict(metadata)
    for key in ("surface_text",):
        if key in value and value[key] is not None:
            edge[key] = value[key]
    return edge


def _parse_strength_vector(value: Any) -> tuple[int, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, int):
        return (max(value, 0),)
    if isinstance(value, str):
        return tuple(max(int(part.strip()), 0) for part in value.replace(";", ",").split(",") if part.strip())
    if isinstance(value, (list, tuple)):
        return tuple(max(int(part), 0) for part in value)
    raise ValueError(f"Некорректный strength_vector: {value!r}")


def _parse_layer_targets(value: Any) -> dict[str, list[str]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Некорректный layer_targets: {value!r}")
    result: dict[str, list[str]] = {}
    for layer, concepts in value.items():
        if isinstance(concepts, str):
            result[str(layer)] = [concepts]
        elif isinstance(concepts, (list, tuple)):
            result[str(layer)] = [str(concept) for concept in concepts]
        else:
            raise ValueError(f"Некорректные layer_targets[{layer!r}]: {concepts!r}")
    return result


def _parse_concept_labels(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Некорректный concept_labels: {value!r}")
    return {str(concept): str(label) for concept, label in value.items() if str(label).strip()}
