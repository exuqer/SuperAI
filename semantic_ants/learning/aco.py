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
from semantic_ants.generation import MiniTransformerSpeechModule
from semantic_ants.learning.checkpoint import Checkpoint, CheckpointStore


TOKEN_RE = re.compile(r"[\wа-яА-ЯёЁ]+", re.UNICODE)


@dataclass(frozen=True)
class Experience:
    stimulus: str
    target_concepts: list[str] = field(default_factory=list)
    accepted_answer: str = ""
    rejected_answers: list[str] = field(default_factory=list)
    positive_edges: list[tuple[str, str, str]] = field(default_factory=list)
    negative_concepts: list[str] = field(default_factory=list)
    reward: float | None = None
    lang: str = "auto"
    modality: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Experience":
        stimulus = str(data.get("stimulus") or data.get("text") or "")
        if not stimulus:
            raise ValueError("Experience требует stimulus или text")
        reward = data.get("reward")
        return cls(
            stimulus=stimulus,
            target_concepts=[str(value) for value in data.get("target_concepts", [])],
            accepted_answer=str(data.get("accepted_answer") or data.get("target_response") or ""),
            rejected_answers=[str(value) for value in data.get("rejected_answers", [])],
            positive_edges=[_parse_edge(value) for value in data.get("positive_edges", [])],
            negative_concepts=[str(value) for value in data.get("negative_concepts", [])],
            reward=float(reward) if reward is not None else None,
            lang=str(data.get("lang", "auto")),
            modality=str(data.get("modality", "text")),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stimulus": self.stimulus,
            "target_concepts": self.target_concepts,
            "accepted_answer": self.accepted_answer,
            "rejected_answers": self.rejected_answers,
            "positive_edges": [list(edge) for edge in self.positive_edges],
            "negative_concepts": self.negative_concepts,
            "reward": self.reward,
            "lang": self.lang,
            "modality": self.modality,
            "metadata": self.metadata,
        }


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
        semantic_score = self._semantic_score(experience.target_concepts, thought.active_concepts, thought.confidence)
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
                "target_concepts": experience.target_concepts,
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
        concept_labels = [_label_from_uri(value) for value in experience.target_concepts]
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
        speech: MiniTransformerSpeechModule | None = None,
    ) -> None:
        self.engine = engine
        self.store = store
        self.judge = judge or Judge()
        self.speech = speech or MiniTransformerSpeechModule()

    def learn_file(self, path: str | Path, epochs: int = 1) -> ACOTrainingReport:
        items = self._read_jsonl(path)
        report = ACOTrainingReport(examples=len(items), epochs=epochs)
        for _ in range(epochs):
            for index, item in enumerate(items, start=1):
                try:
                    self.learn_experience(Experience.from_dict(item), report=report)
                except (KeyError, TypeError, ValueError) as exc:
                    report.errors.append(f"{path}:{index}: {exc}")
        self.store.save(self.engine.checkpoint)
        return report

    def learn_experience(self, experience: Experience, report: ACOTrainingReport | None = None) -> RewardSignal:
        result = self.engine.analyze(
            experience.stimulus,
            lang=experience.lang,
            persist_result=False,
            mode="graph",
        )
        thought = SemanticThought.from_result(result)
        semantic_prompt = thought.to_prompt()
        candidates = self.speech.generate(
            semantic_prompt,
            self.engine.checkpoint,
            fallback=result.response,
            count=3,
        )
        if experience.accepted_answer:
            candidates.insert(0, experience.accepted_answer)
        candidates.extend(experience.rejected_answers)
        best_answer, signal = self.judge.rank(experience, thought, candidates)
        positive = signal.total >= 0.5 or bool(experience.accepted_answer or experience.positive_edges)
        negative = signal.total < 0.2 or bool(experience.rejected_answers or experience.negative_concepts)
        if positive:
            self._reinforce(experience, thought, result.routes, best_answer, signal, report)
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

    def _reinforce(
        self,
        experience: Experience,
        thought: SemanticThought,
        routes: list[AntRoute],
        answer: str,
        signal: RewardSignal,
        report: ACOTrainingReport | None,
    ) -> None:
        checkpoint: Checkpoint = self.engine.checkpoint
        target = set(experience.target_concepts)
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
        for concept in [*experience.target_concepts, *thought.active_concepts]:
            checkpoint.reinforce_concept(concept, amount=amount)
        for start, relation, end in experience.positive_edges:
            checkpoint.add_custom_edge(start, end, relation=relation, weight=max(1.0, amount))
            if checkpoint.add_learned_bridge(start, relation, end, weight=max(1.0, amount), confirmed=True):
                if report:
                    report.learned_bridges += 1
            checkpoint.reinforce_edge(start, relation, end, amount=amount + 0.3)
            if report:
                report.reinforced_edges += 1
        if not experience.positive_edges and len(experience.target_concepts) >= 2:
            for left, right in zip(experience.target_concepts, experience.target_concepts[1:]):
                if checkpoint.add_learned_bridge(left, "LearnedBridge", right, weight=0.4, confirmed=True):
                    if report:
                        report.learned_bridges += 1
        accepted = experience.accepted_answer or answer
        if accepted:
            concepts = experience.target_concepts or thought.active_concepts
            self.speech.train_pair(
                experience.stimulus,
                thought.to_prompt(),
                concepts,
                accepted,
                checkpoint,
                reward=max(signal.total, 0.1),
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
        for concept in experience.negative_concepts:
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
    if isinstance(value, dict):
        return (
            str(value["start"]),
            str(value.get("relation", "LearnedRelatedTo")),
            str(value["end"]),
        )
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            return str(value[0]), "LearnedRelatedTo", str(value[1])
        if len(value) == 3:
            return str(value[0]), str(value[1]), str(value[2])
    raise ValueError(f"Некорректное positive_edges значение: {value!r}")


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
