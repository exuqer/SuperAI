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
        context_plane = context_key or ""
        rng = random.Random(self._seed_for(context_plane))
        routes: list[AntRoute] = []
        for ant_id in range(self.config.ant_count):
            start = starts[ant_id % len(starts)]
            route = self._walk(ant_id, start, graph, checkpoint, rng, context_plane=context_plane)
            routes.append(route)
        return sorted(routes, key=lambda route: route.total_score, reverse=True)

    def _walk(
        self,
        ant_id: int,
        start: str,
        graph: SemanticGraph,
        checkpoint: Checkpoint,
        rng: random.Random,
        *,
        context_plane: str,
    ) -> AntRoute:
        current = start
        current_layer = self._starting_layer(graph, start)
        steps: list[AntStep] = []
        visited = {start}
        total_score = 0.0
        budgets = list(self.config.strength_vector) if self.config.strength_vector else None
        max_steps = sum(max(value, 0) for value in budgets) if budgets is not None else self.config.max_depth
        for _ in range(max_steps):
            candidates = self._candidate_edges(graph.neighbors(current), visited, budgets, current_layer)
            if not candidates:
                break
            edge, pheromone, score, layer_pheromone, projection_shift, next_layer, from_layer, to_layer = self._choose_edge(
                candidates,
                checkpoint,
                rng,
                current_layer=current_layer,
                context_plane=context_plane,
            )
            remaining_strength = self._consume_strength([current_layer, to_layer], budgets)
            steps.append(
                AntStep(
                    start=edge.start,
                    end=edge.end,
                    relation=edge.relation,
                    edge_weight=edge.weight,
                    pheromone=pheromone,
                    score=score,
                    source=edge.source,
                    layer=next_layer,
                    from_layer=from_layer,
                    to_layer=to_layer,
                    context_plane=context_plane,
                    layer_pheromone=layer_pheromone,
                    projection_shift=projection_shift,
                    distance=edge.distance,
                    remaining_strength=remaining_strength,
                    edge_type=edge.edge_type,
                )
            )
            total_score += score
            current = edge.end
            current_layer = next_layer
            visited.add(current)
        return AntRoute(ant_id=ant_id, start=start, steps=steps, total_score=total_score)

    def _candidate_edges(
        self,
        edges: list[SemanticEdge],
        visited: set[str],
        budgets: list[int] | None,
        current_layer: int,
    ) -> list[SemanticEdge]:
        unvisited = [edge for edge in edges if edge.end not in visited]
        if budgets is None:
            return unvisited
        allowed = [edge for edge in unvisited if self._edge_allowed(edge, budgets, current_layer)]
        return allowed

    def _choose_edge(
        self,
        candidates: list[SemanticEdge],
        checkpoint: Checkpoint,
        rng: random.Random,
        *,
        current_layer: int,
        context_plane: str,
    ) -> tuple[SemanticEdge, float, float, float, float, int, int, int]:
        scored = []
        for edge in candidates:
            pheromone = checkpoint.pheromone_for(edge)
            concept_pheromone = checkpoint.concept_pheromone_for(edge.end)
            penalty = checkpoint.penalty_for(edge.end)
            distance = max(float(edge.distance), 0.01)
            from_layer = self._from_layer(edge, current_layer)
            to_layer = self._to_layer(edge, current_layer)
            edge_layer_pheromone = checkpoint.edge_layer_pheromone_for(
                edge,
                from_layer=from_layer,
                to_layer=to_layer,
                context_plane=context_plane,
            )
            concept_layer_pheromone = checkpoint.concept_layer_pheromone_for(edge.end, to_layer, context_plane)
            layer_pheromone = max(edge_layer_pheromone * concept_layer_pheromone, 0.01)
            transition_bonus = 1.0 + max(edge.weight - 1.0, 0.0) * 0.05
            score = max(edge.weight, 0.01) * pheromone * concept_pheromone * layer_pheromone * penalty * transition_bonus / distance
            scored.append((edge, pheromone, max(score, 0.0001), layer_pheromone, abs(to_layer - from_layer), from_layer, to_layer))
        if rng.random() < self.config.exploration:
            edge, pheromone, score, layer_pheromone, projection_shift, from_layer, to_layer = rng.choice(scored)
            return edge, pheromone, score, layer_pheromone, projection_shift, to_layer, from_layer, to_layer
        total = sum(score for _, _, score, *_ in scored)
        cursor = rng.random() * total
        acc = 0.0
        for edge, pheromone, score, layer_pheromone, projection_shift, from_layer, to_layer in scored:
            acc += score
            if acc >= cursor:
                return edge, pheromone, score, layer_pheromone, projection_shift, to_layer, from_layer, to_layer
        edge, pheromone, score, layer_pheromone, projection_shift, from_layer, to_layer = scored[-1]
        return edge, pheromone, score, layer_pheromone, projection_shift, to_layer, from_layer, to_layer

    def _seed_for(self, context_key: str) -> int:
        digest = hashlib.sha256(context_key.encode("utf-8")).hexdigest()
        return self.config.seed + int(digest[:8], 16)

    def _starting_layer(self, graph: SemanticGraph, start: str) -> int:
        node = graph.nodes.get(start)
        layers = list(getattr(node, "active_layers", []) or getattr(node, "layers", []) or [])
        for layer in layers:
            if layer >= 0:
                return layer
        if layers:
            return layers[0]
        return 0

    def _edge_allowed(self, edge: SemanticEdge, budgets: list[int] | None, current_layer: int) -> bool:
        if budgets is None:
            return True
        layers = {self._to_layer(edge, current_layer), edge.layer}
        layers.discard(None)
        return any(self._has_strength(layer, budgets) for layer in layers if layer is not None)

    def _from_layer(self, edge: SemanticEdge, current_layer: int) -> int:
        if edge.from_layer is not None:
            return edge.from_layer
        return current_layer

    def _to_layer(self, edge: SemanticEdge, current_layer: int) -> int:
        if edge.to_layer is not None:
            return edge.to_layer
        return edge.layer

    def _has_strength(self, layer: int, budgets: list[int] | None) -> bool:
        if budgets is None:
            return True
        if layer < 0 or layer >= len(budgets):
            return False
        return budgets[layer] > 0

    def _consume_strength(self, layers: list[int], budgets: list[int] | None) -> int | None:
        if budgets is None:
            return None
        remaining: int | None = None
        for layer in dict.fromkeys(layer for layer in layers if layer >= 0):
            if layer >= len(budgets):
                continue
            budgets[layer] = max(budgets[layer] - 1, 0)
            remaining = budgets[layer] if remaining is None else min(remaining, budgets[layer])
        return remaining
