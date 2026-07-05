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
    strength_vector: tuple[int, ...] = ()


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
        budgets = list(self.config.strength_vector) if self.config.strength_vector else None
        max_steps = sum(max(value, 0) for value in budgets) if budgets is not None else self.config.max_depth
        for _ in range(max_steps):
            candidates = self._candidate_edges(graph.neighbors(current), visited, budgets, allow_layer_fallback=not steps)
            if not candidates:
                break
            edge, pheromone, score = self._choose_edge(candidates, checkpoint, rng)
            remaining_strength = self._consume_strength(edge.layer, budgets)
            steps.append(
                AntStep(
                    start=edge.start,
                    end=edge.end,
                    relation=edge.relation,
                    edge_weight=edge.weight,
                    pheromone=pheromone,
                    score=score,
                    source=edge.source,
                    layer=edge.layer,
                    distance=edge.distance,
                    remaining_strength=remaining_strength,
                    edge_type=edge.edge_type,
                )
            )
            total_score += score
            current = edge.end
            visited.add(current)
        return AntRoute(ant_id=ant_id, start=start, steps=steps, total_score=total_score)

    def _candidate_edges(
        self,
        edges: list[SemanticEdge],
        visited: set[str],
        budgets: list[int] | None,
        allow_layer_fallback: bool,
    ) -> list[SemanticEdge]:
        unvisited = [edge for edge in edges if edge.end not in visited]
        if budgets is None:
            return unvisited
        allowed = [edge for edge in unvisited if self._has_strength(edge.layer, budgets)]
        if allowed:
            return allowed
        # A short strength_vector such as "3" means "prefer top-layer search".
        # If the current word has no edge in that layer, keep the ants moving
        # through ordinary semantic edges instead of returning empty routes.
        return unvisited if allow_layer_fallback else []

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
            distance = max(float(edge.distance), 0.01)
            score = max(edge.weight, 0.01) * pheromone * concept_pheromone * penalty / distance
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

    def _has_strength(self, layer: int, budgets: list[int] | None) -> bool:
        if budgets is None:
            return True
        if layer < 0 or layer >= len(budgets):
            return False
        return budgets[layer] > 0

    def _consume_strength(self, layer: int, budgets: list[int] | None) -> int | None:
        if budgets is None:
            return None
        if layer < 0 or layer >= len(budgets):
            return None
        budgets[layer] = max(budgets[layer] - 1, 0)
        return budgets[layer]
