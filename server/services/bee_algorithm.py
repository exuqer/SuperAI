"""Artificial Bee Colony context foraging over the persistent 2D field."""

from __future__ import annotations

import asyncio
import math
import random
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set, Tuple

from server.repositories.cloud_repository import CloudPlacementRepository, CloudRepository, LayerRepository, SpaceRepository


EventCallback = Callable[[str, Dict[str, Any]], Awaitable[None]]


@dataclass
class BeeAlgorithmConfig:
    scout_bees: int = 12
    employed_bees: int = 16
    onlooker_bees: int = 20
    min_iterations: int = 4
    max_iterations: int = 16
    bee_capacity: float = 0.5
    scout_search_radius: float = 80.0
    employed_search_radius: float = 24.0
    onlooker_search_radius: float = 16.0
    source_acceptance_threshold: float = 0.18
    harvest_threshold: float = 0.28
    abandonment_limit: int = 6
    maximum_harvests_per_source: int = 8
    selection_temperature_start: float = 1.0
    selection_temperature_end: float = 0.35
    saturation_rate: float = 0.12
    saturation_decay: float = 0.90
    current_message_region_ratio: float = 0.40
    intersection_region_ratio: float = 0.25
    dialogue_region_ratio: float = 0.20
    random_region_ratio: float = 0.15
    weight_strength: float = 0.20
    weight_query_alignment: float = 0.28
    weight_dialogue_alignment: float = 0.16
    weight_intersection: float = 0.18
    weight_novelty: float = 0.12
    weight_scene_support: float = 0.06
    penalty_redundancy: float = 0.14
    penalty_common_word: float = 0.04
    penalty_contradiction: float = 0.10


@dataclass
class FieldSample:
    world_x: float
    world_y: float
    composition: Dict[str, float]
    cloud_ids: Dict[str, int]
    total_strength: float
    scene_support: float = 0.0


@dataclass
class ForagingGoal:
    current_message_id: str
    active_cloud_ids: List[int]
    active_scene_ids: List[int]
    query_composition: Dict[str, float]
    dialogue_composition: Dict[str, float]
    current_message_weight: float = 1.0
    previous_message_weights: Dict[str, float] = field(default_factory=dict)
    desired_layers: Set[str] = field(default_factory=lambda: {"word_form", "lexeme", "concept", "scene"})
    novelty_preference: float = 0.35
    intersection_preference: float = 0.45

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["desired_layers"] = sorted(self.desired_layers)
        return data


@dataclass
class NectarPayload:
    source_id: str
    strength: float
    composition: Dict[str, float]
    world_x: float
    world_y: float
    iteration: int


@dataclass
class NectarSource:
    id: str
    world_x: float
    world_y: float
    sample: FieldSample
    fitness: float
    query_alignment: float
    dialogue_alignment: float
    novelty: float
    overlap_value: float
    visits: int = 0
    successful_updates: int = 0
    failed_updates: int = 0
    trial_count: int = 0
    source_message_ids: Set[str] = field(default_factory=set)
    discovered_by_bee_id: str = ""
    state: str = "DISCOVERED"
    created_iteration: int = 0
    updated_iteration: int = 0
    harvested_strength: float = 0.0
    saturation: float = 0.0
    harvest_count: int = 0

    @property
    def effective_fitness(self) -> float:
        return max(0.0, self.fitness * (1.0 - min(0.98, self.saturation)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "x": self.world_x,
            "y": self.world_y,
            "fitness": round(self.fitness, 6),
            "effective_fitness": round(self.effective_fitness, 6),
            "query": round(self.query_alignment, 6),
            "dialogue": round(self.dialogue_alignment, 6),
            "novelty": round(self.novelty, 6),
            "intersection": round(self.overlap_value, 6),
            "state": self.state,
            "visits": self.visits,
            "trial_count": self.trial_count,
            "saturation": round(self.saturation, 6),
            "composition": self.sample.composition,
        }


@dataclass
class Bee:
    id: str
    role: str
    x: float
    y: float
    source_id: Optional[str] = None
    capacity: float = 0.5
    status: str = "idle"

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "role": self.role, "x": self.x, "y": self.y, "source_id": self.source_id, "status": self.status}


@dataclass
class BeeDance:
    bee_id: str
    source_id: str
    fitness: float
    direction: Tuple[float, float]
    source_distance: float
    nectar_strength: float
    composition: Dict[str, float]
    confidence: float
    iteration: int


@dataclass
class SwarmState:
    session_id: str
    turn_index: int
    iteration: int = 0
    scouts: Dict[str, Bee] = field(default_factory=dict)
    employed_bees: Dict[str, Bee] = field(default_factory=dict)
    onlookers: Dict[str, Bee] = field(default_factory=dict)
    sources: Dict[str, NectarSource] = field(default_factory=dict)
    dances: List[BeeDance] = field(default_factory=list)
    visited: List[Tuple[float, float]] = field(default_factory=list)
    collected_payloads: List[NectarPayload] = field(default_factory=list)
    best_source_id: Optional[str] = None
    average_fitness: float = 0.0
    diversity: float = 0.0


class FieldSampler:
    """Samples local mixtures from global placements without copying cloud mass."""

    def __init__(self, desired_layers: Iterable[str]):
        self.cloud_repo = CloudRepository()
        self.placement_repo = CloudPlacementRepository()
        self.space_repo = SpaceRepository()
        self.layer_repo = LayerRepository()
        self.placements: List[Tuple[Any, Any, str]] = []
        self.by_name: Dict[str, List[Tuple[Any, Any, str]]] = {}
        self._load(set(desired_layers))

    def _load(self, layers: Set[str]) -> None:
        for layer_name in layers:
            layer = self.layer_repo.get_by_name(layer_name)
            if not layer:
                continue
            space = self.space_repo.get_global_space(layer.id)
            if not space:
                continue
            for placement in self.placement_repo.get_by_space(space.id):
                cloud = self.cloud_repo.get_by_id(placement.cloud_id)
                if not cloud:
                    continue
                item = (placement, cloud, layer_name)
                self.placements.append(item)
                self.by_name.setdefault(cloud.canonical_name.casefold(), []).append(item)

    def positions_for_labels(self, labels: Iterable[str]) -> List[Tuple[float, float]]:
        positions: List[Tuple[float, float]] = []
        for label in labels:
            for placement, _, _ in self.by_name.get(label.casefold(), []):
                positions.append((placement.x, placement.y))
        return positions

    def context_areas(self, max_items: int = 900) -> List[Dict[str, Any]]:
        """Return the global concept meadows used as the swarm's search field."""
        layer_priority = {"concept": 0, "scene": 1, "lexeme": 2, "word_form": 3}
        areas: List[Dict[str, Any]] = []
        for placement, cloud, layer_name in self.placements:
            layer_scale = {"concept": 1.25, "scene": 1.55, "lexeme": 0.92, "word_form": 0.72}.get(layer_name, 0.8)
            areas.append({
                "id": f"{layer_name}-{cloud.id}-{placement.id or 0}",
                "token": cloud.canonical_name,
                "x": round(float(placement.x), 3),
                "y": round(float(placement.y), 3),
                "radius": round(max(12.0, float(placement.radius) * layer_scale), 3),
                "strength": round(max(0.05, float(placement.density)) * max(0.05, float(placement.activation) + 0.12), 6),
                "density": round(float(placement.density), 6),
                "layer": layer_name,
            })
        areas.sort(key=lambda item: (layer_priority.get(item["layer"], 9), -item["strength"], item["token"]))
        return areas[:max_items]

    def sample(self, x: float, y: float, radius: float = 90.0) -> FieldSample:
        values: Dict[str, float] = {}
        cloud_ids: Dict[str, int] = {}
        scene_support = 0.0
        for placement, cloud, layer_name in self.placements:
            distance = math.hypot(x - placement.x, y - placement.y)
            sigma = max(12.0, radius * 0.45 + placement.radius * 0.55)
            influence = math.exp(-((distance / sigma) ** 2) / 2.0)
            strength = influence * max(0.05, placement.density) * max(0.05, placement.activation + 0.12)
            if strength <= 0.005:
                continue
            key = cloud.canonical_name.casefold()
            values[key] = values.get(key, 0.0) + strength
            cloud_ids[key] = cloud.id
            if layer_name == "scene":
                scene_support = max(scene_support, min(1.0, strength))
        total = sum(values.values())
        if total:
            composition = {key: value / total for key, value in values.items()}
        else:
            composition = {}
        return FieldSample(x, y, composition, cloud_ids, total, scene_support)

    def seed_positions(self, goal: ForagingGoal, rng: random.Random) -> List[Tuple[float, float]]:
        query = self.positions_for_labels(goal.query_composition.keys())
        dialogue = self.positions_for_labels(goal.dialogue_composition.keys())
        seeds: List[Tuple[float, float]] = []
        seeds.extend(query)
        for left, right in zip(query[::2], query[1::2]):
            seeds.append(((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0))
        seeds.extend(dialogue)
        if not self.placements:
            bounds = (0.0, 1000.0, 0.0, 700.0)
        else:
            xs = [p.x for p, _, _ in self.placements]
            ys = [p.y for p, _, _ in self.placements]
            bounds = (min(xs) - 80, max(xs) + 80, min(ys) - 80, max(ys) + 80)
        while len(seeds) < 64:
            if seeds and rng.random() < 0.75:
                base = rng.choice(seeds)
                seeds.append((base[0] + rng.uniform(-80, 80), base[1] + rng.uniform(-80, 80)))
            else:
                seeds.append((rng.uniform(bounds[0], bounds[1]), rng.uniform(bounds[2], bounds[3])))
        return seeds


def _cosine(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    keys = set(left) | set(right)
    numerator = sum(left.get(key, 0.0) * right.get(key, 0.0) for key in keys)
    denominator = math.sqrt(sum(value * value for value in left.values())) * math.sqrt(sum(value * value for value in right.values()))
    return numerator / denominator if denominator else 0.0


class BeeSwarm:
    def __init__(self, goal: ForagingGoal, session_id: str, turn_index: int, config: Optional[BeeAlgorithmConfig] = None):
        self.goal = goal
        self.config = config or BeeAlgorithmConfig()
        self.rng = random.Random(f"{session_id}:{turn_index}:{goal.current_message_id}")
        self.state = SwarmState(session_id=session_id, turn_index=turn_index)
        self.sampler = FieldSampler(goal.desired_layers)
        self.positions = self.sampler.seed_positions(goal, self.rng)
        self.position_index = 0

    def _next_position(self, radius: float) -> Tuple[float, float]:
        if self.position_index >= len(self.positions):
            self.positions.extend(self.sampler.seed_positions(self.goal, self.rng))
        base = self.positions[self.position_index]
        self.position_index += 1
        return base[0] + self.rng.uniform(-radius, radius), base[1] + self.rng.uniform(-radius, radius)

    def _fitness(self, sample: FieldSample) -> Tuple[float, Dict[str, float]]:
        query = _cosine(self.goal.query_composition, sample.composition)
        dialogue = _cosine(self.goal.dialogue_composition, sample.composition)
        if not self.goal.query_composition and not self.goal.dialogue_composition:
            return 0.0, {"query": 0.0, "dialogue": 0.0, "intersection": 0.0, "novelty": 0.0}
        relevant = [sample.composition.get(key, 0.0) for key in self.goal.query_composition]
        intersection = 0.0
        if len(relevant) >= 2:
            present = [value for value in relevant if value > 0.03]
            if present:
                intersection = min(1.0, (len(present) / len(relevant)) * (sum(present) / len(present)) * 2.0)
        novelty = 1.0
        if self.state.visited:
            nearest = min(math.hypot(sample.world_x - x, sample.world_y - y) for x, y in self.state.visited)
            novelty = min(1.0, nearest / 120.0)
        redundancy = 0.0
        for source in self.state.sources.values():
            redundancy = max(redundancy, _cosine(source.sample.composition, sample.composition))
        strength = min(1.0, sample.total_strength / 2.0)
        score = (
            self.config.weight_strength * strength
            + self.config.weight_query_alignment * query
            + self.config.weight_dialogue_alignment * dialogue
            + self.config.weight_intersection * intersection
            + self.config.weight_novelty * novelty
            + self.config.weight_scene_support * sample.scene_support
            - self.config.penalty_redundancy * redundancy
        )
        return max(0.0, min(1.0, score)), {"query": query, "dialogue": dialogue, "intersection": intersection, "novelty": novelty}

    def _new_source(self, bee: Bee, iteration: int) -> Optional[NectarSource]:
        x, y = self._next_position(self.config.scout_search_radius)
        sample = self.sampler.sample(x, y, self.config.scout_search_radius)
        fitness, parts = self._fitness(sample)
        self.state.visited.append((x, y))
        if fitness < self.config.source_acceptance_threshold:
            return None
        source = NectarSource(
            id=f"source-{uuid.uuid4().hex[:10]}", world_x=x, world_y=y, sample=sample,
            fitness=fitness, query_alignment=parts["query"], dialogue_alignment=parts["dialogue"],
            novelty=parts["novelty"], overlap_value=parts["intersection"], discovered_by_bee_id=bee.id,
            source_message_ids={self.goal.current_message_id}, created_iteration=iteration, updated_iteration=iteration,
        )
        self.state.sources[source.id] = source
        bee.source_id = source.id
        bee.status = "assigned"
        return source

    async def _emit(self, callback: EventCallback, event_type: str, payload: Dict[str, Any]) -> None:
        await callback(event_type, payload)

    async def run(self, callback: EventCallback) -> SwarmState:
        await self._emit(callback, "swarm_started", {
            "goal": self.goal.to_dict(),
            "config": asdict(self.config),
            "context_areas": self.sampler.context_areas(),
        })
        for index in range(self.config.scout_bees):
            bee = Bee(f"scout-{index}", "scout", 0.0, 0.0, capacity=self.config.bee_capacity)
            self.state.scouts[bee.id] = bee
            await self._emit(callback, "scout_spawned", {"bee": bee.to_dict()})
        for index in range(self.config.employed_bees):
            bee = Bee(f"employed-{index}", "employed", 0.0, 0.0, capacity=self.config.bee_capacity)
            self.state.employed_bees[bee.id] = bee
        for index in range(self.config.onlooker_bees):
            bee = Bee(f"onlooker-{index}", "onlooker", 0.0, 0.0, capacity=self.config.bee_capacity)
            self.state.onlookers[bee.id] = bee
        for iteration in range(1, self.config.max_iterations + 1):
            self.state.iteration = iteration
            self.state.dances = []
            for bee in list(self.state.employed_bees.values()):
                if not bee.source_id or bee.source_id not in self.state.sources:
                    continue
                source = self.state.sources[bee.source_id]
                neighbour = self.rng.choice(list(self.state.sources.values())) if len(self.state.sources) > 1 else source
                phi_x, phi_y = self.rng.uniform(-1, 1), self.rng.uniform(-1, 1)
                candidate_x = source.world_x + phi_x * (source.world_x - neighbour.world_x)
                candidate_y = source.world_y + phi_y * (source.world_y - neighbour.world_y)
                max_step = self.config.employed_search_radius
                candidate_x = source.world_x + max(-max_step, min(max_step, candidate_x - source.world_x))
                candidate_y = source.world_y + max(-max_step, min(max_step, candidate_y - source.world_y))
                sample = self.sampler.sample(candidate_x, candidate_y, max_step)
                candidate_fitness, parts = self._fitness(sample)
                source.visits += 1
                bee.x, bee.y, bee.status = candidate_x, candidate_y, "searching"
                await self._emit(callback, "source_candidate_tested", {"bee": bee.to_dict(), "source": source.to_dict(), "candidate": {"x": candidate_x, "y": candidate_y, "fitness": candidate_fitness}})
                if candidate_fitness > source.fitness:
                    source.world_x, source.world_y, source.sample = candidate_x, candidate_y, sample
                    source.fitness = candidate_fitness
                    source.query_alignment, source.dialogue_alignment = parts["query"], parts["dialogue"]
                    source.novelty, source.overlap_value = parts["novelty"], parts["intersection"]
                    source.successful_updates += 1
                    source.trial_count = 0
                    source.state = "ACTIVE"
                    await self._emit(callback, "source_improved", {"source": source.to_dict(), "bee": bee.to_dict()})
                else:
                    source.failed_updates += 1
                    source.trial_count += 1
                    if source.trial_count >= max(1, self.config.abandonment_limit // 2):
                        source.state = "WEAK"
                    await self._emit(callback, "source_not_improved", {"source": source.to_dict(), "bee": bee.to_dict()})
                source.saturation *= self.config.saturation_decay
                if self._should_harvest(source):
                    await self._harvest(source, bee, callback)
                self.state.dances.append(BeeDance(bee.id, source.id, source.fitness, (source.world_x - bee.x, source.world_y - bee.y), math.hypot(source.world_x - bee.x, source.world_y - bee.y), source.sample.total_strength, source.sample.composition, min(1.0, source.fitness + 0.2), iteration))
                await self._emit(callback, "dance_published", {"bee_id": bee.id, "source": source.to_dict(), "iteration": iteration})

            active_sources = [source for source in self.state.sources.values() if source.state != "ABANDONED"]
            if active_sources:
                temperature = self.config.selection_temperature_start + (self.config.selection_temperature_end - self.config.selection_temperature_start) * ((iteration - 1) / max(1, self.config.max_iterations - 1))
                dance_scores = {}
                for dance in self.state.dances:
                    dance_scores[dance.source_id] = max(dance_scores.get(dance.source_id, 0.0), dance.fitness)
                weights = [math.exp((source.effective_fitness * 0.7 + dance_scores.get(source.id, source.effective_fitness) * 0.3) / max(0.05, temperature)) for source in active_sources]
                for bee in self.state.onlookers.values():
                    source = self.rng.choices(active_sources, weights=weights, k=1)[0]
                    bee.source_id = source.id
                    bee.x, bee.y, bee.status = source.world_x, source.world_y, "selecting"
                    await self._emit(callback, "onlooker_source_selected", {"bee": bee.to_dict(), "source": source.to_dict(), "probability": weights[active_sources.index(source)] / sum(weights)})
                    candidate_x = source.world_x + self.rng.uniform(-self.config.onlooker_search_radius, self.config.onlooker_search_radius)
                    candidate_y = source.world_y + self.rng.uniform(-self.config.onlooker_search_radius, self.config.onlooker_search_radius)
                    sample = self.sampler.sample(candidate_x, candidate_y, self.config.onlooker_search_radius)
                    score, parts = self._fitness(sample)
                    source.visits += 1
                    if score > source.fitness:
                        source.world_x, source.world_y, source.sample, source.fitness = candidate_x, candidate_y, sample, score
                        source.query_alignment, source.dialogue_alignment = parts["query"], parts["dialogue"]
                        source.novelty, source.overlap_value, source.trial_count = parts["novelty"], parts["intersection"], 0
                        source.successful_updates += 1
                        await self._emit(callback, "source_improved", {"source": source.to_dict(), "bee": bee.to_dict()})
                    else:
                        source.trial_count += 1
                    if self._should_harvest(source):
                        await self._harvest(source, bee, callback)

            for source in list(self.state.sources.values()):
                if source.trial_count >= self.config.abandonment_limit:
                    source.state = "ABANDONED"
                    await self._emit(callback, "source_abandoned", {"source": source.to_dict()})
                    for bee in list(self.state.employed_bees.values()):
                        if bee.source_id == source.id:
                            bee.source_id, bee.role, bee.status = None, "scout", "idle"
                            self.state.scouts[bee.id] = bee
                            self.state.employed_bees.pop(bee.id, None)
                            await self._emit(callback, "bee_became_scout", {"bee": bee.to_dict()})
            for bee in list(self.state.scouts.values()):
                if bee.source_id:
                    continue
                source = self._new_source(bee, iteration)
                if source:
                    self.state.employed_bees[bee.id] = Bee(bee.id, "employed", source.world_x, source.world_y, source.id, bee.capacity, "assigned")
                    self.state.scouts.pop(bee.id, None)
                    await self._emit(callback, "source_discovered", {"source": source.to_dict(), "bee": bee.to_dict()})
                else:
                    await self._emit(callback, "source_rejected", {"bee": bee.to_dict(), "x": bee.x, "y": bee.y})

            active_sources = [source for source in self.state.sources.values() if source.state != "ABANDONED"]
            self.state.best_source_id = max(active_sources, key=lambda item: item.effective_fitness).id if active_sources else None
            self.state.average_fitness = sum(source.effective_fitness for source in active_sources) / len(active_sources) if active_sources else 0.0
            compositions = [set(source.sample.composition) for source in active_sources]
            union = set().union(*compositions) if compositions else set()
            self.state.diversity = len(union) / max(1, len(active_sources))
            await self._emit(callback, "swarm_iteration_completed", {
                "iteration": iteration,
                "metrics": {
                    "best_fitness": self.state.sources[self.state.best_source_id].effective_fitness if self.state.best_source_id else 0.0,
                    "average_fitness": self.state.average_fitness,
                    "diversity": self.state.diversity,
                    "source_count": len(active_sources),
                },
                "sources": [source.to_dict() for source in active_sources],
                "bees": [bee.to_dict() for bee in list(self.state.scouts.values()) + list(self.state.employed_bees.values()) + list(self.state.onlookers.values())],
                "context_areas": self.sampler.context_areas(),
            })
            if iteration >= self.config.min_iterations and active_sources and self.state.average_fitness >= 0.72 and self.state.diversity >= 1.0:
                await self._emit(callback, "swarm_stabilized", {"iteration": iteration, "metrics": {"average_fitness": self.state.average_fitness, "diversity": self.state.diversity}})
                break
            await asyncio.sleep(0)
        await self._emit(callback, "swarm_completed", {"iteration": self.state.iteration, "payload_count": len(self.state.collected_payloads), "sources": [source.to_dict() for source in self.state.sources.values() if source.state != "ABANDONED"]})
        return self.state

    def _should_harvest(self, source: NectarSource) -> bool:
        return source.fitness >= self.config.harvest_threshold and source.sample.total_strength > 0 and source.harvest_count < self.config.maximum_harvests_per_source and source.effective_fitness >= self.config.harvest_threshold

    async def _harvest(self, source: NectarSource, bee: Bee, callback: EventCallback) -> None:
        strength = min(self.config.bee_capacity, source.sample.total_strength)
        payload = NectarPayload(source.id, strength, dict(source.sample.composition), source.world_x, source.world_y, self.state.iteration)
        source.harvest_count += 1
        source.harvested_strength += strength
        source.saturation = min(0.98, source.saturation + strength * self.config.saturation_rate)
        source.state = "EXPLOITED"
        self.state.collected_payloads.append(payload)
        await self._emit(callback, "nectar_collected", {"payload": asdict(payload), "source": source.to_dict(), "bee": bee.to_dict()})
