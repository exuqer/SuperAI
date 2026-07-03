from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from semantic_ants.core.models import SemanticEdge, edge_key


@dataclass
class Checkpoint:
    """Обучаемый слой поверх внешнего графа знаний."""

    version: int = 2
    pheromones: dict[str, float] = field(default_factory=dict)
    concept_pheromones: dict[str, float] = field(default_factory=dict)
    suppressed_concepts: dict[str, float] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    custom_edges: list[dict[str, Any]] = field(default_factory=list)
    response_memory: dict[str, dict[str, Any]] = field(default_factory=dict)
    negative_memory: list[dict[str, Any]] = field(default_factory=list)
    experiences: list[dict[str, Any]] = field(default_factory=list)
    learned_bridges: list[dict[str, Any]] = field(default_factory=list)
    accepted_answers: list[dict[str, Any]] = field(default_factory=list)
    route_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    mini_generator: dict[str, Any] = field(default_factory=dict)
    last_result_id: str | None = None
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    examples_seen: int = 0
    seed: int = 42

    def pheromone_for(self, edge: SemanticEdge) -> float:
        return max(float(self.pheromones.get(edge.key, 1.0)), 0.01)

    def concept_pheromone_for(self, concept_uri: str) -> float:
        return max(float(self.concept_pheromones.get(concept_uri, 1.0)), 0.01)

    def penalty_for(self, concept_uri: str) -> float:
        penalty = max(float(self.suppressed_concepts.get(concept_uri, 0.0)), 0.0)
        return 1.0 / (1.0 + penalty)

    def reinforce_edge(self, start: str, relation: str, end: str, amount: float = 0.2) -> None:
        key = edge_key(start, relation, end)
        self.pheromones[key] = min(float(self.pheromones.get(key, 1.0)) + amount, 25.0)
        stat = self.route_stats.setdefault(key, {"positive": 0, "negative": 0})
        stat["positive"] = int(stat.get("positive", 0)) + 1
        self.reinforce_concept(start, amount=amount * 0.25)
        self.reinforce_concept(end, amount=amount * 0.5)

    def penalize_edge(self, start: str, relation: str, end: str, amount: float = 0.2) -> None:
        key = edge_key(start, relation, end)
        self.pheromones[key] = max(float(self.pheromones.get(key, 1.0)) - amount, 0.05)
        stat = self.route_stats.setdefault(key, {"positive": 0, "negative": 0})
        stat["negative"] = int(stat.get("negative", 0)) + 1
        self.penalize_concept(end, amount=amount * 0.5)

    def reinforce_concept(self, concept_uri: str, amount: float = 0.2) -> None:
        if not concept_uri:
            return
        self.concept_pheromones[concept_uri] = min(
            float(self.concept_pheromones.get(concept_uri, 1.0)) + amount,
            25.0,
        )
        if concept_uri in self.suppressed_concepts:
            self.suppressed_concepts[concept_uri] = max(
                float(self.suppressed_concepts.get(concept_uri, 0.0)) - amount,
                0.0,
            )

    def penalize_concept(self, concept_uri: str, amount: float = 0.2) -> None:
        if not concept_uri:
            return
        self.concept_pheromones[concept_uri] = max(
            float(self.concept_pheromones.get(concept_uri, 1.0)) - amount,
            0.05,
        )

    def suppress_concept(self, concept_uri: str, amount: float = 0.5) -> None:
        self.suppressed_concepts[concept_uri] = min(
            float(self.suppressed_concepts.get(concept_uri, 0.0)) + amount,
            25.0,
        )
        self.penalize_concept(concept_uri, amount=amount * 0.25)

    def remember_response(self, concepts: list[str], response: str, amount: float = 1.0) -> None:
        if not concepts or not response:
            return
        key = concept_set_key(concepts)
        item = self.response_memory.setdefault(key, {"response": response, "weight": 0.0})
        item["response"] = response
        item["weight"] = float(item.get("weight", 0.0)) + amount

    def remember_accepted_answer(
        self,
        stimulus: str,
        semantic_prompt: str,
        concepts: list[str],
        answer: str,
        reward: float = 1.0,
        limit: int = 500,
    ) -> None:
        if not answer:
            return
        item = {
            "stimulus": stimulus,
            "semantic_prompt": semantic_prompt,
            "concepts": list(dict.fromkeys(concepts)),
            "answer": answer,
            "reward": float(reward),
            "created_at": time.time(),
        }
        self.accepted_answers.append(item)
        self.accepted_answers.sort(key=lambda value: float(value.get("reward", 0.0)), reverse=True)
        del self.accepted_answers[limit:]
        self.remember_response(item["concepts"], answer, amount=max(float(reward), 0.1))

    def remember_negative(
        self,
        stimulus: str,
        semantic_prompt: str,
        concepts: list[str],
        answer: str,
        reason: str = "",
        limit: int = 500,
    ) -> None:
        item = {
            "stimulus": stimulus,
            "semantic_prompt": semantic_prompt,
            "concepts": list(dict.fromkeys(concepts)),
            "answer": answer,
            "reason": reason,
            "created_at": time.time(),
        }
        self.negative_memory.append(item)
        del self.negative_memory[:-limit]

    def remember_experience(self, experience: dict[str, Any], limit: int = 2000) -> None:
        self.experiences.append({**experience, "stored_at": time.time()})
        del self.experiences[:-limit]

    def remember_result(self, result: dict[str, Any], limit: int = 50) -> None:
        result_id = str(result["result_id"])
        self.last_result_id = result_id
        self.results[result_id] = result
        while len(self.results) > limit:
            oldest = next(iter(self.results))
            del self.results[oldest]

    def learned_edges(self) -> list[SemanticEdge]:
        edges: list[SemanticEdge] = []
        seen: set[str] = set()
        for raw in [*self.custom_edges, *self.learned_bridges]:
            try:
                edge = SemanticEdge(
                    start=str(raw["start"]),
                    end=str(raw["end"]),
                    relation=str(raw.get("relation", "LearnedRelatedTo")),
                    weight=float(raw.get("weight", 1.0)),
                    source="learned",
                    surface_text=raw.get("surface_text"),
                    metadata=dict(raw.get("metadata", {})),
                )
                if edge.key in seen:
                    continue
                seen.add(edge.key)
                edges.append(edge)
            except (KeyError, TypeError, ValueError):
                continue
        return edges

    def add_custom_edge(
        self,
        start: str,
        end: str,
        relation: str = "LearnedRelatedTo",
        weight: float = 1.0,
    ) -> None:
        candidate = {
            "start": start,
            "end": end,
            "relation": relation,
            "weight": weight,
        }
        if candidate not in self.custom_edges:
            self.custom_edges.append(candidate)

    def add_learned_bridge(
        self,
        start: str,
        relation: str,
        end: str,
        weight: float = 0.3,
        confirmed: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        bridge = {
            "start": start,
            "relation": relation,
            "end": end,
            "weight": weight,
            "confirmed": confirmed,
            "metadata": dict(metadata or {}),
        }
        for existing in self.learned_bridges:
            if (
                existing.get("start") == start
                and existing.get("relation") == relation
                and existing.get("end") == end
            ):
                if existing.get("confirmed"):
                    return False
                existing["weight"] = max(float(existing.get("weight", 0.0)), weight)
                existing["confirmed"] = bool(existing.get("confirmed")) or confirmed
                existing["metadata"] = {**dict(existing.get("metadata", {})), **bridge["metadata"]}
                return False
        self.learned_bridges.append(bridge)
        self.add_custom_edge(start, end, relation=relation, weight=weight)
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "pheromones": self.pheromones,
            "concept_pheromones": self.concept_pheromones,
            "suppressed_concepts": self.suppressed_concepts,
            "aliases": self.aliases,
            "custom_edges": self.custom_edges,
            "response_memory": self.response_memory,
            "negative_memory": self.negative_memory,
            "experiences": self.experiences,
            "learned_bridges": self.learned_bridges,
            "accepted_answers": self.accepted_answers,
            "route_stats": self.route_stats,
            "metadata": self.metadata,
            "mini_generator": self.mini_generator,
            "last_result_id": self.last_result_id,
            "results": self.results,
            "examples_seen": self.examples_seen,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        return cls(
            version=max(int(data.get("version", 1)), 2),
            pheromones=dict(data.get("pheromones", {})),
            concept_pheromones=dict(data.get("concept_pheromones", {})),
            suppressed_concepts=dict(data.get("suppressed_concepts", {})),
            aliases=dict(data.get("aliases", {})),
            custom_edges=list(data.get("custom_edges", [])),
            response_memory=dict(data.get("response_memory", {})),
            negative_memory=list(data.get("negative_memory", [])),
            experiences=list(data.get("experiences", [])),
            learned_bridges=list(data.get("learned_bridges", [])),
            accepted_answers=list(data.get("accepted_answers", [])),
            route_stats=dict(data.get("route_stats", {})),
            metadata=dict(data.get("metadata", {})),
            mini_generator=dict(data.get("mini_generator", {})),
            last_result_id=data.get("last_result_id"),
            results=dict(data.get("results", {})),
            examples_seen=int(data.get("examples_seen", 0)),
            seed=int(data.get("seed", 42)),
        )


class CheckpointStore:
    """Читает и сохраняет checkpoint в JSON."""

    def __init__(self, path: Path | str = ".semantic_ants/checkpoints/model.json") -> None:
        self.path = Path(path)

    def load(self) -> Checkpoint:
        if not self.path.exists():
            return Checkpoint()
        with self.path.open("r", encoding="utf-8") as handle:
            return Checkpoint.from_dict(json.load(handle))

    def save(self, checkpoint: Checkpoint) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(checkpoint.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
        _replace_with_retry(tmp, self.path)

    def export(self, destination: Path | str) -> None:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            shutil.copyfile(self.path, target)
        else:
            self.save(Checkpoint())
            shutil.copyfile(self.path, target)


def concept_set_key(concepts: list[str]) -> str:
    return "|".join(sorted(dict.fromkeys(concepts)))


def _replace_with_retry(tmp: Path, target: Path, attempts: int = 20) -> None:
    for index in range(attempts):
        try:
            tmp.replace(target)
            return
        except PermissionError:
            if index == attempts - 1:
                raise
            time.sleep(0.05 * (index + 1))
