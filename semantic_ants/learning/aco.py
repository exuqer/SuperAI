from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from semantic_ants.ants import AntColony, AntConfig
from semantic_ants.core.graph import SemanticGraph
from semantic_ants.core.models import AntRoute, SemanticEdge, SemanticResult
from semantic_ants.generation import TorchDialogueNavigator
from semantic_ants.learning.checkpoint import Checkpoint, CheckpointStore


TOKEN_RE = re.compile(r"[\wа-яА-ЯёЁ]+", re.UNICODE)


@dataclass(frozen=True)
class Experience:
    stimulus: str
    target_concepts: list[str] = field(default_factory=list)
    accepted_answer: str = ""
    rejected_answers: list[str] = field(default_factory=list)
    positive_edges: list[tuple[str, str, str] | dict[str, Any]] = field(default_factory=list)
    negative_concepts: list[str] = field(default_factory=list)
    reward: float | None = None
    lang: str = "auto"
    strength_vector: tuple[int, ...] = ()
    layer_targets: dict[str, list[str]] = field(default_factory=dict)
    concept_labels: dict[str, str] = field(default_factory=dict)
    modality: str = "text"
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Experience":
        stimulus = str(data.get("stimulus") or data.get("text") or data.get("question") or "")
        if not stimulus:
            raise ValueError("Experience требует stimulus, text или question")
        reward = data.get("reward")
        return cls(
            stimulus=stimulus,
            target_concepts=[str(value) for value in data.get("target_concepts", [])],
            accepted_answer=str(
                data.get("accepted_answer")
                or data.get("expected_answer")
                or data.get("target_response")
                or ""
            ),
            rejected_answers=[str(value) for value in data.get("rejected_answers", [])],
            positive_edges=_parse_positive_edges(data.get("positive_edges")),
            negative_concepts=[str(value) for value in data.get("negative_concepts", [])],
            reward=float(reward) if reward is not None else None,
            lang=str(data.get("lang", "auto")),
            strength_vector=_parse_strength_vector(data.get("strength_vector")),
            layer_targets=_parse_layer_targets(data.get("layer_targets")),
            concept_labels=_parse_concept_labels(data.get("concept_labels")),
            modality=str(data.get("modality", "text")),
            history=[dict(value) for value in data.get("history", []) if isinstance(value, dict)],
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stimulus": self.stimulus,
            "target_concepts": self.target_concepts,
            "accepted_answer": self.accepted_answer,
            "rejected_answers": self.rejected_answers,
            "positive_edges": [dict(edge) if isinstance(edge, dict) else list(edge) for edge in self.positive_edges],
            "negative_concepts": self.negative_concepts,
            "reward": self.reward,
            "lang": self.lang,
            "strength_vector": list(self.strength_vector),
            "layer_targets": self.layer_targets,
            "concept_labels": self.concept_labels,
            "modality": self.modality,
            "history": self.history,
            "metadata": self.metadata,
        }

    @property
    def learning_targets(self) -> list[str]:
        values = list(self.target_concepts)
        for concepts in self.layer_targets.values():
            values.extend(concepts)
        return list(dict.fromkeys(values))


@dataclass(frozen=True)
class SemanticThought:
    stimulus: str
    active_concepts: list[str]
    facts: list[str]
    causal_links: list[dict[str, str]]
    confidence: float
    contradictions: list[str]
    answer_plan: list[str]
    routes: list[dict[str, Any]]

    @classmethod
    def from_result(cls, result: SemanticResult) -> "SemanticThought":
        active_concepts = [str(item["uri"]) for item in result.activated_concepts]
        routes = [route.to_dict() for route in result.routes[:8]]
        facts: list[str] = []
        causal_links: list[dict[str, str]] = []
        contradictions: list[str] = []
        for route in result.routes[:8]:
            if len(route.concepts) != len(set(route.concepts)):
                contradictions.append(f"route#{route.ant_id}: repeated concept")
            for step in route.steps:
                fact = f"{step.start} --{step.relation}--> {step.end}"
                if fact not in facts:
                    facts.append(fact)
                if _looks_causal(step.relation):
                    causal_links.append(
                        {
                            "start": step.start,
                            "relation": step.relation,
                            "end": step.end,
                        }
                    )
        top_score = max((route.total_score for route in result.routes), default=0.0)
        total_score = sum(max(route.total_score, 0.0) for route in result.routes[:8])
        confidence = min(1.0, top_score / max(total_score, 1.0) + min(len(active_concepts), 5) * 0.08)
        answer_plan = _answer_plan(active_concepts, facts, contradictions)
        return cls(
            stimulus=result.input_text,
            active_concepts=active_concepts,
            facts=facts[:16],
            causal_links=causal_links[:8],
            confidence=round(confidence, 4),
            contradictions=contradictions[:8],
            answer_plan=answer_plan,
            routes=routes,
        )

    def to_prompt(self, style: str = "concise") -> str:
        route_lines = []
        for route in self.routes[:5]:
            concepts = " -> ".join(map(str, route.get("concepts", [])))
            route_lines.append(f"ant#{route.get('ant_id')} score={route.get('total_score')}: {concepts}")
        return "\n".join(
            [
                f"user_question: {self.stimulus}",
                f"style: {style}",
                f"active_concepts: {', '.join(self.active_concepts)}",
                f"facts: {'; '.join(self.facts[:8])}",
                f"causal_links: {json.dumps(self.causal_links, ensure_ascii=False)}",
                f"confidence: {self.confidence}",
                f"contradictions: {'; '.join(self.contradictions) if self.contradictions else 'none'}",
                f"answer_plan: {'; '.join(self.answer_plan)}",
                "ant_routes:",
                *route_lines,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stimulus": self.stimulus,
            "active_concepts": self.active_concepts,
            "facts": self.facts,
            "causal_links": self.causal_links,
            "confidence": self.confidence,
            "contradictions": self.contradictions,
            "answer_plan": self.answer_plan,
            "routes": self.routes,
            "prompt": self.to_prompt(),
        }


@dataclass(frozen=True)
class RewardSignal:
    semantic_score: float
    answer_score: float
    contradiction_penalty: float
    human_score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return self.semantic_score + self.answer_score + self.human_score - self.contradiction_penalty

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_score": self.semantic_score,
            "answer_score": self.answer_score,
            "contradiction_penalty": self.contradiction_penalty,
            "human_score": self.human_score,
            "total": self.total,
            "details": self.details,
        }


class Judge:
    def rank(
        self,
        experience: Experience,
        thought: SemanticThought,
        candidates: list[str],
    ) -> tuple[str, RewardSignal]:
        values = candidates or [""]
        scored = [(candidate, self.evaluate(experience, thought, candidate)) for candidate in values]
        scored.sort(key=lambda value: value[1].total, reverse=True)
        return scored[0]

    def rank_freeform(self, stimulus: str, thought: SemanticThought, candidates: list[str]) -> tuple[str, RewardSignal]:
        return self.rank(Experience(stimulus=stimulus), thought, candidates)

    def evaluate(self, experience: Experience, thought: SemanticThought, answer: str) -> RewardSignal:
        target_concepts = experience.learning_targets
        semantic_score = self._semantic_score(target_concepts, thought.active_concepts, thought.confidence)
        answer_score = self._answer_score(experience, answer)
        contradiction_penalty = self._logic_penalty(experience, thought, answer)
        human_score = float(experience.reward or 0.0)
        return RewardSignal(
            semantic_score=round(semantic_score, 4),
            answer_score=round(answer_score, 4),
            contradiction_penalty=round(contradiction_penalty, 4),
            human_score=round(human_score, 4),
            details={
                "answer": answer,
                "target_concepts": target_concepts,
                "active_concepts": thought.active_concepts,
            },
        )

    def _semantic_score(self, target_concepts: list[str], active_concepts: list[str], confidence: float) -> float:
        if not target_concepts:
            return max(confidence, 0.1)
        target = set(target_concepts)
        active = set(active_concepts)
        overlap = len(target & active)
        if not overlap:
            return 0.0
        return overlap / len(target)

    def _answer_score(self, experience: Experience, answer: str) -> float:
        if not answer.strip():
            return 0.0
        if _matches_any(answer, experience.rejected_answers):
            return 0.0
        if not experience.accepted_answer:
            return 0.3
        accepted_terms = _terms(experience.accepted_answer)
        answer_terms = _terms(answer)
        if not accepted_terms:
            return 0.3
        lexical = len(accepted_terms & answer_terms) / len(accepted_terms)
        concept_labels = [_label_from_uri(value) for value in experience.learning_targets]
        label_hits = sum(1 for label in concept_labels if label and label.lower() in answer.lower())
        concept_score = label_hits / max(len(concept_labels), 1)
        return min(1.0, lexical * 0.7 + concept_score * 0.3)

    def _logic_penalty(self, experience: Experience, thought: SemanticThought, answer: str) -> float:
        penalty = len(thought.contradictions) * 0.2
        clean = answer.strip().lower()
        if not clean:
            penalty += 1.0
        if _matches_any(answer, experience.rejected_answers):
            penalty += 1.0
        terms = list(_terms(answer))
        if len(terms) >= 4:
            counts = Counter(terms)
            if max(counts.values()) / len(terms) > 0.45:
                penalty += 0.4
        for concept in experience.negative_concepts:
            label = _label_from_uri(concept).lower()
            if label and label in clean:
                penalty += 0.5
        return min(penalty, 3.0)


@dataclass
class ACOTrainingReport:
    examples: int = 0
    epochs: int = 0
    reinforced_edges: int = 0
    evaporated_edges: int = 0
    suppressed_concepts: int = 0
    learned_bridges: int = 0
    accepted_answers: int = 0
    negative_samples: int = 0
    experiences: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "examples": self.examples,
            "epochs": self.epochs,
            "reinforced_edges": self.reinforced_edges,
            "evaporated_edges": self.evaporated_edges,
            "suppressed_concepts": self.suppressed_concepts,
            "learned_bridges": self.learned_bridges,
            "accepted_answers": self.accepted_answers,
            "negative_samples": self.negative_samples,
            "experiences": self.experiences,
            "errors": self.errors,
        }


class ACOTrainer:
    def __init__(
        self,
        engine: Any,
        store: CheckpointStore,
        judge: Judge | None = None,
        speech: TorchDialogueNavigator | None = None,
    ) -> None:
        self.engine = engine
        self.store = store
        self.judge = judge or Judge()
        self.speech = speech or TorchDialogueNavigator()
        self.model_dir = getattr(engine, "model_dir", self.store.path.parent.parent / "models")
        self._deferred_dialogue_pairs: list[dict[str, Any]] = []

    def learn_file(self, path: str | Path, epochs: int = 1) -> ACOTrainingReport:
        items = self._read_jsonl(path)
        report = ACOTrainingReport(examples=len(items), epochs=epochs)
        for _ in range(epochs):
            for index, item in enumerate(items, start=1):
                try:
                    self.learn_experience(Experience.from_dict(item), report=report)
                except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                    report.errors.append(f"{path}:{index}: {exc}")
        self.store.save(self.engine.checkpoint)
        return report

    def learn_dialogue_file(
        self,
        path: str | Path,
        epochs: int = 1,
        batch_size: int = 32,
        max_examples: int | None = None,
        torch_steps: int = 1,
    ) -> ACOTrainingReport:
        items = self._read_jsonl(path)
        if max_examples is not None:
            items = items[:max_examples]
        report = ACOTrainingReport(examples=len(items), epochs=epochs)
        for _ in range(epochs):
            for index, item in enumerate(items, start=1):
                try:
                    self.learn_experience(Experience.from_dict(item), report=report, defer_speech=True)
                except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                    report.errors.append(f"{path}:{index}: {exc}")
                if batch_size > 0 and index % batch_size == 0:
                    self._flush_dialogue_training(torch_steps=torch_steps)
                    self.store.save(self.engine.checkpoint)
            self._flush_dialogue_training(torch_steps=torch_steps)
        self.store.save(self.engine.checkpoint)
        return report

    def learn_experience(
        self,
        experience: Experience,
        report: ACOTrainingReport | None = None,
        defer_speech: bool = False,
    ) -> RewardSignal:
        result = self.engine.analyze(
            experience.stimulus,
            lang=experience.lang,
            persist_result=False,
            mode="graph",
            generate_response=False,
            strength_vector=experience.strength_vector or None,
        )
        thought = SemanticThought.from_result(result)
        semantic_prompt = self.speech.build_prompt(
            input_text=experience.stimulus,
            tokens=result.tokens,
            activated_concepts=result.activated_concepts,
            routes=result.routes,
            checkpoint=self.engine.checkpoint,
            chat_history=experience.history,
        )
        if experience.accepted_answer:
            candidates = [experience.accepted_answer]
        else:
            candidates = self.speech.generate(
                semantic_prompt,
                self.engine.checkpoint,
                model_dir=self.model_dir,
                fallback=result.response,
                count=3,
            )
        candidates.extend(experience.rejected_answers)
        best_answer, signal = self.judge.rank(experience, thought, candidates)
        positive = signal.total >= 0.5 or bool(experience.accepted_answer or experience.positive_edges)
        negative = signal.total < 0.2 or bool(experience.rejected_answers or experience.negative_concepts)
        if positive:
            self._remember_labels(experience)
            self._reinforce(
                experience,
                thought,
                result.routes,
                best_answer,
                signal,
                report,
                semantic_prompt=semantic_prompt,
                defer_speech=defer_speech,
            )
        if negative:
            self._evaporate(experience, thought, result.routes, best_answer, signal, report)
        self.engine.checkpoint.remember_experience(
            {
                **experience.to_dict(),
                "thought": thought.to_dict(),
                "selected_answer": best_answer,
                "reward_signal": signal.to_dict(),
            }
        )
        self.engine.checkpoint.examples_seen += 1
        if report:
            report.experiences += 1
        return signal

    def _remember_labels(self, experience: Experience) -> None:
        for concept, label in experience.concept_labels.items():
            self.engine.checkpoint.remember_concept_label(concept, label)

    def dream(self, steps: int = 100) -> dict[str, Any]:
        checkpoint: Checkpoint = self.engine.checkpoint
        graph = SemanticGraph()
        graph.add_edges(checkpoint.learned_edges(), include_reverse=True)
        starts = sorted(graph.nodes)
        rng = random.Random(checkpoint.seed + steps + len(checkpoint.learned_bridges))
        report = {"steps": steps, "bridges": 0, "rejected": 0}
        if not starts:
            return report
        for index in range(steps):
            start = rng.choice(starts)
            routes = AntColony(
                AntConfig(
                    ant_count=1,
                    max_depth=3,
                    seed=rng.randint(1, 1_000_000),
                    exploration=0.5,
                )
            ).search(graph, [start], checkpoint, context_key=f"dream:{index}:{start}")
            route = routes[0] if routes else None
            if not route or len(route.concepts) < 3:
                report["rejected"] += 1
                continue
            end = route.concepts[-1]
            if start == end or _confirmed_bridge_exists(checkpoint, start, end):
                report["rejected"] += 1
                continue
            if len(route.concepts) != len(set(route.concepts)):
                report["rejected"] += 1
                continue
            added = checkpoint.add_learned_bridge(
                start=start,
                relation="DreamAssociatedWith",
                end=end,
                weight=0.12,
                confirmed=False,
                metadata={"dream": True, "route": route.to_dict()},
            )
            if added:
                report["bridges"] += 1
                graph.add_edge(
                    SemanticEdge(
                        start=start,
                        relation="DreamAssociatedWith",
                        end=end,
                        weight=0.12,
                        source="dream",
                        metadata={"dream": True},
                    ),
                    include_reverse=True,
                )
            else:
                report["rejected"] += 1
        self.store.save(checkpoint)
        return report

    def inspect_memory(self) -> dict[str, Any]:
        checkpoint: Checkpoint = self.engine.checkpoint
        top_edges = sorted(checkpoint.pheromones.items(), key=lambda value: value[1], reverse=True)[:10]
        top_concepts = sorted(checkpoint.concept_pheromones.items(), key=lambda value: value[1], reverse=True)[:10]
        return {
            "version": checkpoint.version,
            "examples_seen": checkpoint.examples_seen,
            "pheromones": len(checkpoint.pheromones),
            "concept_pheromones": len(checkpoint.concept_pheromones),
            "negative_memory": len(checkpoint.negative_memory),
            "experiences": len(checkpoint.experiences),
            "learned_bridges": len(checkpoint.learned_bridges),
            "accepted_answers": len(checkpoint.accepted_answers),
            "response_memory": len(checkpoint.response_memory),
            "top_edges": top_edges,
            "top_concepts": top_concepts,
        }

    def _flush_dialogue_training(self, torch_steps: int = 1) -> None:
        if not self._deferred_dialogue_pairs:
            return
        pairs = self._deferred_dialogue_pairs
        self._deferred_dialogue_pairs = []
        self.speech.train_pairs(
            pairs,
            self.engine.checkpoint,
            model_dir=self.model_dir,
            steps=torch_steps,
        )

    def _reinforce(
        self,
        experience: Experience,
        thought: SemanticThought,
        routes: list[AntRoute],
        answer: str,
        signal: RewardSignal,
        report: ACOTrainingReport | None,
        semantic_prompt: str | None = None,
        defer_speech: bool = False,
    ) -> None:
        checkpoint: Checkpoint = self.engine.checkpoint
        learning_targets = experience.learning_targets
        target = set(learning_targets)
        selected_routes = [route for route in routes if target & set(route.concepts)] if target else []
        if not selected_routes:
            selected_routes = routes[:3]
        amount = max(0.1, min(signal.total, 3.0) * 0.18)
        for route in selected_routes:
            for step in route.steps:
                step_amount = amount
                if target and step.start not in target and step.end not in target:
                    step_amount *= 0.5
                checkpoint.reinforce_edge(step.start, step.relation, step.end, amount=step_amount)
                if report:
                    report.reinforced_edges += 1
        for concept in [*learning_targets, *thought.active_concepts]:
            checkpoint.reinforce_concept(concept, amount=amount)
        for edge_value in experience.positive_edges:
            if isinstance(edge_value, dict):
                edge_start = str(edge_value["start"])
                edge_relation = str(edge_value.get("relation", "LearnedRelatedTo"))
                edge_end = str(edge_value["end"])
                edge_weight = max(float(edge_value.get("weight", 1.0)), amount)
                edge_layer = int(edge_value.get("layer", 1))
                edge_distance = float(edge_value.get("distance", 1.0))
                edge_type = str(edge_value.get("edge_type", "semantic"))
                raw_metadata = edge_value.get("metadata", {})
                edge_metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else dict(raw_metadata or {})
                checkpoint.add_custom_edge(
                    edge_start,
                    edge_end,
                    relation=edge_relation,
                    weight=edge_weight,
                    layer=edge_layer,
                    distance=edge_distance,
                    edge_type=edge_type,
                    metadata=edge_metadata,
                )
                if checkpoint.add_learned_bridge(
                    edge_start,
                    edge_relation,
                    edge_end,
                    weight=edge_weight,
                    confirmed=True,
                    metadata={
                        **edge_metadata,
                        "layer": edge_layer,
                        "distance": edge_distance,
                        "edge_type": edge_type,
                    },
                ):
                    if report:
                        report.learned_bridges += 1
                checkpoint.reinforce_edge(edge_start, edge_relation, edge_end, amount=edge_weight + 0.3)
                if report:
                    report.reinforced_edges += 1
                continue
            start, relation, end = edge_value
            checkpoint.add_custom_edge(start, end, relation=relation, weight=max(1.0, amount))
            if checkpoint.add_learned_bridge(start, relation, end, weight=max(1.0, amount), confirmed=True):
                if report:
                    report.learned_bridges += 1
            checkpoint.reinforce_edge(start, relation, end, amount=amount + 0.3)
            if report:
                report.reinforced_edges += 1
        route_starts = list(dict.fromkeys(route.start for route in routes))
        for layer_key, concepts in experience.layer_targets.items():
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
                        weight=max(1.0, amount),
                        layer=layer,
                        distance=1.0,
                        edge_type=edge_type,
                        metadata={"layer_target": layer_key},
                    )
                    checkpoint.reinforce_edge(start, relation, concept, amount=amount + 0.2)
                    if report:
                        report.reinforced_edges += 1
        if not experience.positive_edges and len(learning_targets) >= 2:
            for left, right in zip(learning_targets, learning_targets[1:]):
                if checkpoint.add_learned_bridge(left, "LearnedBridge", right, weight=0.4, confirmed=True):
                    if report:
                        report.learned_bridges += 1
        accepted = experience.accepted_answer or answer
        if accepted:
            concepts = learning_targets or thought.active_concepts
            prompt = semantic_prompt or thought.to_prompt()
            if defer_speech:
                self._deferred_dialogue_pairs.append(
                    {
                        "stimulus": experience.stimulus,
                        "prompt": prompt,
                        "concepts": concepts,
                        "answer": accepted,
                        "reward": max(signal.total, 0.1),
                    }
                )
            else:
                self.speech.train_pair(
                    experience.stimulus,
                    prompt,
                    concepts,
                    accepted,
                    checkpoint,
                    reward=max(signal.total, 0.1),
                    model_dir=self.model_dir,
                )
            if report:
                report.accepted_answers += 1

    def _evaporate(
        self,
        experience: Experience,
        thought: SemanticThought,
        routes: list[AntRoute],
        answer: str,
        signal: RewardSignal,
        report: ACOTrainingReport | None,
    ) -> None:
        checkpoint: Checkpoint = self.engine.checkpoint
        amount = max(0.1, min(abs(signal.total) + signal.contradiction_penalty + 0.2, 2.0) * 0.15)
        negative = set(experience.negative_concepts)
        selected_routes = [route for route in routes if negative & set(route.concepts)] if negative else []
        if not selected_routes:
            selected_routes = routes[:3]
        for route in selected_routes[:3]:
            for step in route.steps:
                checkpoint.penalize_edge(step.start, step.relation, step.end, amount=amount)
                if report:
                    report.evaporated_edges += 1
        route_starts = list(dict.fromkeys(route.start for route in routes))
        for concept in experience.negative_concepts:
            for start in route_starts:
                checkpoint.penalize_edge(start, "RelatedTo", concept, amount=amount)
                if report:
                    report.evaporated_edges += 1
            checkpoint.suppress_concept(concept, amount=0.75)
            if report:
                report.suppressed_concepts += 1
        for rejected in experience.rejected_answers:
            self.speech.train_negative(
                experience.stimulus,
                thought.to_prompt(),
                thought.active_concepts,
                rejected,
                checkpoint,
                reason="dataset rejected answer",
            )
            if report:
                report.negative_samples += 1
        if answer and (signal.total < 0.2 or _matches_any(answer, experience.rejected_answers)):
            self.speech.train_negative(
                experience.stimulus,
                thought.to_prompt(),
                thought.active_concepts,
                answer,
                checkpoint,
                reason="low reward selected answer",
            )
            if report:
                report.negative_samples += 1

    def _read_jsonl(self, path: str | Path) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                items.append(json.loads(stripped))
        return items


def _parse_edge(value: Any) -> tuple[str, str, str]:
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
    return [_parse_positive_edge(item) for item in value]


def _parse_positive_edge(value: Any) -> tuple[str, str, str] | dict[str, Any]:
    if isinstance(value, dict):
        return _parse_positive_edge_dict(value)
    return _parse_edge(value)


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
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        return tuple(max(int(part), 0) for part in parts)
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


def _looks_causal(relation: str) -> bool:
    lower = relation.lower()
    return any(value in lower for value in ("cause", "affect", "because", "trigger", "canparticipate"))


def _answer_plan(active_concepts: list[str], facts: list[str], contradictions: list[str]) -> list[str]:
    if contradictions:
        return ["проверить противоречия", "ответить осторожно"]
    if not active_concepts:
        return ["сообщить, что связей недостаточно"]
    plan = [f"использовать главный концепт {_label_from_uri(active_concepts[0])}"]
    if facts:
        plan.append("объяснить сильнейшие связи маршрута")
    if len(active_concepts) > 1:
        plan.append("связать соседние активные концепты")
    return plan


def _terms(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def _matches_any(answer: str, rejected_answers: list[str]) -> bool:
    clean = " ".join(answer.lower().split())
    return any(clean == " ".join(value.lower().split()) for value in rejected_answers)


def _label_from_uri(uri: str) -> str:
    return uri.rstrip("/").split("/")[-1].replace("_", " ")


def _confirmed_bridge_exists(checkpoint: Checkpoint, start: str, end: str) -> bool:
    for bridge in checkpoint.learned_bridges:
        if not bridge.get("confirmed"):
            continue
        if bridge.get("start") == start and bridge.get("end") == end:
            return True
    return False
