from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from semantic_ants.core.graph import SemanticGraph
from semantic_ants.core.models import AntRoute, AntStep, SemanticEdge
from semantic_ants.learning.checkpoint import Checkpoint


@dataclass(frozen=True)
class AntConfig:
    ant_count: int = 32
    max_depth: int = 4
    seed: int = 42
    exploration: float = 0.12


class AntColony:
    """Вероятностный поиск смысловых маршрутов с учетом феромонов."""

    def __init__(self, config: AntConfig | None = None) -> None:
        self.config = config or AntConfig()

    def search(
        self,
        graph: SemanticGraph,
        starts: list[str],
        checkpoint: Checkpoint,
        context_key: str = "",
    ) -> list[AntRoute]:
        if not starts:
            return []
        rng = random.Random(self._seed_for(context_key))
        routes: list[AntRoute] = []
        for ant_id in range(self.config.ant_count):
            start = starts[ant_id % len(starts)]
            route = self._walk(ant_id, start, graph, checkpoint, rng)
            routes.append(route)
        return sorted(routes, key=lambda route: route.total_score, reverse=True)

    def _walk(
        self,
        ant_id: int,
        start: str,
        graph: SemanticGraph,
        checkpoint: Checkpoint,
        rng: random.Random,
    ) -> AntRoute:
        current = start
        steps: list[AntStep] = []
        visited = {start}
        total_score = 0.0
        for _ in range(self.config.max_depth):
            candidates = [edge for edge in graph.neighbors(current) if edge.end not in visited]
            if not candidates:
                break
            edge, pheromone, score = self._choose_edge(candidates, checkpoint, rng)
            steps.append(
                AntStep(
                    start=edge.start,
                    end=edge.end,
                    relation=edge.relation,
                    edge_weight=edge.weight,
                    pheromone=pheromone,
                    score=score,
                    source=edge.source,
                )
            )
            total_score += score
            current = edge.end
            visited.add(current)
        return AntRoute(ant_id=ant_id, start=start, steps=steps, total_score=total_score)

    def _choose_edge(
        self,
        candidates: list[SemanticEdge],
        checkpoint: Checkpoint,
        rng: random.Random,
    ) -> tuple[SemanticEdge, float, float]:
        scored = []
        for edge in candidates:
            pheromone = checkpoint.pheromone_for(edge)
            concept_pheromone = checkpoint.concept_pheromone_for(edge.end)
            penalty = checkpoint.penalty_for(edge.end)
            score = max(edge.weight, 0.01) * pheromone * concept_pheromone * penalty
            scored.append((edge, pheromone, max(score, 0.0001)))
        if rng.random() < self.config.exploration:
            edge, pheromone, score = rng.choice(scored)
            return edge, pheromone, score
        total = sum(score for _, _, score in scored)
        cursor = rng.random() * total
        acc = 0.0
        for edge, pheromone, score in scored:
            acc += score
            if acc >= cursor:
                return edge, pheromone, score
        return scored[-1]

    def _seed_for(self, context_key: str) -> int:
        digest = hashlib.sha256(context_key.encode("utf-8")).hexdigest()
        return self.config.seed + int(digest[:8], 16)
