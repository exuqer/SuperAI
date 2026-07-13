"""Relation-free 2D gradient-field physics."""

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple


Vector = List[float]


@dataclass
class ConceptState:
    id: int
    token: str
    position: Vector = field(default_factory=lambda: [0.0, 0.0])
    mass: float = 1.0
    radius: float = 0.0
    activation: float = 0.0
    velocity: Vector = field(default_factory=lambda: [0.0, 0.0])
    phase: float = 0.0

    def __post_init__(self) -> None:
        self.position = [float(self.position[0]), float(self.position[1])]
        self.velocity = [float(self.velocity[0]), float(self.velocity[1])]
        self.mass = max(0.001, float(self.mass))
        self.radius = self.radius or concept_radius(self.mass)

    @property
    def word(self) -> str:
        return self.token

    def distance_to(self, other: "ConceptState") -> float:
        return math.hypot(self.position[0] - other.position[0], self.position[1] - other.position[1])

    def apply_force(self, force: Sequence[float]) -> None:
        self.velocity[0] += finite(force[0]) / self.mass
        self.velocity[1] += finite(force[1]) / self.mass

    def step(self, config: "PhysicsConfig") -> None:
        self.velocity[0] = finite(self.velocity[0] * config.damping, 0.0)
        self.velocity[1] = finite(self.velocity[1] * config.damping, 0.0)
        self.position[0] = finite(self.position[0] + self.velocity[0], config.width / 2)
        self.position[1] = finite(self.position[1] + self.velocity[1], config.height / 2)
        self.position[0] = max(config.boundary_margin, min(config.width - config.boundary_margin, self.position[0]))
        self.position[1] = max(config.boundary_margin, min(config.height - config.boundary_margin, self.position[1]))


WordState = ConceptState


@dataclass
class PhysicsConfig:
    width: float = 1600.0
    height: float = 1000.0
    gravity_constant: float = 3200.0
    gravity_softening: float = 64.0
    gravity_max_force: float = 6.0
    repulsion_max_force: float = 10.0
    phrase_impulse_factor: float = 0.025
    phrase_impulse_max: float = 1.5
    ordered_learning_rate: float = 0.018
    context_memory: float = 0.82
    damping: float = 0.82
    steps: int = 60
    boundary_margin: float = 32.0


def finite(value: float, fallback: float = 0.0) -> float:
    return float(value) if math.isfinite(value) else fallback


def concept_radius(mass: float) -> float:
    return min(250.0, 22.0 + 12.0 * math.sqrt(max(0.001, float(mass))))


def field_strength(mass: float, distance: float) -> float:
    distance = max(0.0, float(distance))
    return finite(float(mass) / (distance * distance + 64.0))


def compute_center(concepts: Iterable[ConceptState]) -> Tuple[float, float]:
    items = list(concepts)
    if not items:
        return 800.0, 500.0
    weights = [max(0.001, concept.mass * max(concept.activation, 0.0)) for concept in items]
    if not any(concept.activation > 0 for concept in items):
        weights = [max(0.001, concept.mass) for concept in items]
    total = sum(weights)
    return (
        finite(sum(concept.position[0] * weight for concept, weight in zip(items, weights)) / total, 800.0),
        finite(sum(concept.position[1] * weight for concept, weight in zip(items, weights)) / total, 500.0),
    )


def place_new_concepts(
    new_concepts: Sequence[ConceptState],
    existing_concepts: Sequence[ConceptState],
    config: PhysicsConfig,
) -> None:
    if not new_concepts:
        return
    center_x, center_y = compute_center(existing_concepts)
    radius = 150.0 if not existing_concepts else min(260.0, max(100.0, sum(
        math.hypot(concept.position[0] - center_x, concept.position[1] - center_y)
        for concept in existing_concepts
    ) / len(existing_concepts)))
    count = len(new_concepts)
    for index, concept in enumerate(new_concepts):
        angle = (2.0 * math.pi * index) / max(1, count)
        concept.position = [
            max(config.boundary_margin, min(config.width - config.boundary_margin, center_x + radius * math.cos(angle))),
            max(config.boundary_margin, min(config.height - config.boundary_margin, center_y + radius * math.sin(angle))),
        ]


def apply_gradient_force(first: ConceptState, second: ConceptState, config: PhysicsConfig) -> Tuple[float, float]:
    dx = second.position[0] - first.position[0]
    dy = second.position[1] - first.position[1]
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        return 0.0, 0.0
    magnitude = config.gravity_constant * math.sqrt(first.mass * second.mass)
    magnitude /= distance * distance + config.gravity_softening
    magnitude = min(config.gravity_max_force, max(0.0, finite(magnitude)))
    return magnitude * dx / distance, magnitude * dy / distance


def apply_repulsion(first: ConceptState, second: ConceptState, config: PhysicsConfig) -> Tuple[float, float]:
    dx = second.position[0] - first.position[0]
    dy = second.position[1] - first.position[1]
    distance = math.hypot(dx, dy)
    threshold = max(1.0, first.radius + second.radius)
    if distance >= threshold:
        return 0.0, 0.0
    if distance < 1e-6:
        angle = ((first.id * 37 + second.id * 17) % 360) * math.pi / 180.0
        return config.repulsion_max_force * math.cos(angle), config.repulsion_max_force * math.sin(angle)
    magnitude = config.repulsion_max_force * (1.0 - distance / threshold)
    return -magnitude * dx / distance, -magnitude * dy / distance


def positional_weight(distance: int) -> float:
    if distance <= 1:
        return 1.0
    if distance == 2:
        return 0.5
    return 0.1


def apply_sentence_impulses(
    concepts: Dict[str, ConceptState],
    sentences: Sequence[Sequence[str]],
    config: PhysicsConfig,
) -> None:
    for sentence in sentences:
        for left_index in range(len(sentence)):
            left = concepts.get(sentence[left_index])
            if left is None:
                continue
            for right_index in range(left_index + 1, len(sentence)):
                right = concepts.get(sentence[right_index])
                if right is None or right is left:
                    continue
                dx = right.position[0] - left.position[0]
                dy = right.position[1] - left.position[1]
                distance = math.hypot(dx, dy)
                if distance < 1e-6:
                    continue
                magnitude = config.phrase_impulse_factor * positional_weight(right_index - left_index)
                magnitude *= min(config.phrase_impulse_max, max(0.0, distance - left.radius - right.radius))
                force = [magnitude * dx / distance, magnitude * dy / distance]
                left.apply_force(force)
                right.apply_force([-force[0], -force[1]])


def apply_ordered_trajectory(
    concepts: Dict[str, ConceptState],
    sentences: Sequence[Sequence[str]],
    context_position: Tuple[float, float],
    config: PhysicsConfig,
) -> Tuple[float, float]:
    context = [float(context_position[0]), float(context_position[1])]
    for sentence in sentences:
        for token in sentence:
            concept = concepts.get(token)
            if concept is None:
                continue
            delta_x = context[0] - concept.position[0]
            delta_y = context[1] - concept.position[1]
            concept.position[0] += finite(config.ordered_learning_rate * delta_x)
            concept.position[1] += finite(config.ordered_learning_rate * delta_y)
            context[0] = config.context_memory * context[0] + (1.0 - config.context_memory) * concept.position[0]
            context[1] = config.context_memory * context[1] + (1.0 - config.context_memory) * concept.position[1]
    return context[0], context[1]


def run_physics_step(
    concepts: List[ConceptState],
    sentences: Sequence[Sequence[str]],
    config: PhysicsConfig,
) -> None:
    for concept in concepts:
        concept.velocity = [0.0, 0.0]
        concept.radius = concept_radius(concept.mass)

    by_token = {concept.token: concept for concept in concepts}
    for index, first in enumerate(concepts):
        for second in concepts[index + 1:]:
            attraction = apply_gradient_force(first, second, config)
            repulsion = apply_repulsion(first, second, config)
            first.apply_force([attraction[0] + repulsion[0], attraction[1] + repulsion[1]])
            second.apply_force([-attraction[0] - repulsion[0], -attraction[1] - repulsion[1]])

    apply_sentence_impulses(by_token, sentences, config)
    for concept in concepts:
        concept.step(config)


def run_simulation(
    concepts: List[ConceptState],
    sentences: Sequence[Sequence[str]],
    config: PhysicsConfig,
    context_position: Tuple[float, float] | None = None,
) -> Tuple[float, float]:
    by_token = {concept.token: concept for concept in concepts}
    context = context_position or compute_center(concepts)
    context = apply_ordered_trajectory(by_token, sentences, context, config)
    for _ in range(config.steps):
        run_physics_step(concepts, sentences, config)
    return context
