"""2D Gravitational Physics Simulation for word space physics"""
import math
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class WordState:
    """State of a word in the 2D space"""
    word: str
    mass: float = 1.0
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    gravity: float = 1.0

    def distance_to(self, other: "WordState") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return math.hypot(dx, dy)

    def apply_force(self, fx: float, fy: float):
        self.vx += fx / self.mass
        self.vy += fy / self.mass

    def step(self, damping: float = 0.82, bounds: Tuple[float, float] = (1000, 700)):
        # Apply damping
        self.vx *= damping
        self.vy *= damping
        # Update position
        self.x += self.vx
        self.y += self.vy
        # Clamp to bounds
        width, height = bounds
        margin = 50
        self.x = max(margin, min(width - margin, self.x))
        self.y = max(margin, min(height - margin, self.y))


@dataclass
class PhysicsConfig:
    """Physics simulation configuration"""
    width: float = 1000.0
    height: float = 700.0
    gravity_constant: float = 6000.0
    gravity_min_distance: float = 32.0
    gravity_min_force: float = 0.02
    gravity_max_force: float = 12.0
    phrase_impulse_factor: float = 0.015
    phrase_impulse_max: float = 7.0
    phrase_target_distance: float = 110.0
    repulsion_distance: float = 88.0
    repulsion_max_force: float = 20.0
    damping: float = 0.82
    steps: int = 60
    boundary_margin: float = 50.0


def compute_center(words: List[WordState]) -> Tuple[float, float]:
    """Compute center of mass for a list of words."""
    if not words:
        return 500.0, 350.0  # Default center
    total_mass = sum(w.mass for w in words)
    if total_mass == 0:
        return 500.0, 350.0
    cx = sum(w.x * w.mass for w in words) / total_mass
    cy = sum(w.y * w.mass for w in words) / total_mass
    return cx, cy


def place_new_words_around_center(
    new_words: List[WordState],
    existing_words: List[WordState],
    config: PhysicsConfig,
) -> None:
    """Place new words around the center of known words or map center."""
    if existing_words:
        cx, cy = compute_center(existing_words)
    else:
        cx, cy = config.width / 2, config.height / 2

    # Place new words at average distance from center
    if existing_words:
        avg_dist = sum(math.hypot(w.x - cx, w.y - cy) for w in existing_words) / len(existing_words)
        avg_dist = max(avg_dist, 80.0)
    else:
        avg_dist = 150.0

    for i, word in enumerate(new_words):
        angle = (2 * math.pi * i) / max(len(new_words), 1) + random.uniform(-0.2, 0.2)
        word.x = cx + avg_dist * math.cos(angle)
        word.y = cy + avg_dist * math.sin(angle)


def apply_gravity(word1: WordState, word2: WordState, config: PhysicsConfig) -> Tuple[float, float]:
    """Compute gravitational force between two words. Returns (fx, fy) on word1 from word2."""
    dx = word2.x - word1.x
    dy = word2.y - word1.y
    dist = math.hypot(dx, dy)
    if dist < config.gravity_min_distance:
        dist = config.gravity_min_distance
    if dist == 0:
        return 0.0, 0.0

    # Gravity is deliberately observable: frequency supplies the baseline and
    # stable neighbors increase the body's pull.
    force = config.gravity_constant * math.sqrt(max(word1.gravity, 0.1) * max(word2.gravity, 0.1)) / (dist * dist)
    force = max(config.gravity_min_force, min(config.gravity_max_force, force))

    fx = force * dx / dist
    fy = force * dy / dist
    return fx, fy


def apply_phrase_impulse(word1: WordState, word2: WordState, config: PhysicsConfig) -> Tuple[float, float]:
    """Compute phrase impulse (attraction) between words in same phrase."""
    dx = word2.x - word1.x
    dy = word2.y - word1.y
    dist = math.hypot(dx, dy)
    if dist == 0:
        return 0.0, 0.0

    # Impulse = min(7, 0.015 * m1 * m2 * max(dist - 110, 0))
    impulse_magnitude = config.phrase_impulse_factor * word1.mass * word2.mass * max(dist - config.phrase_target_distance, 0)
    impulse_magnitude = min(config.phrase_impulse_max, impulse_magnitude)

    fx = impulse_magnitude * dx / dist
    fy = impulse_magnitude * dy / dist
    return fx, fy


def apply_repulsion(word1: WordState, word2: WordState, config: PhysicsConfig) -> Tuple[float, float]:
    """Compute repulsion force when words are too close."""
    dx = word2.x - word1.x
    dy = word2.y - word1.y
    dist = math.hypot(dx, dy)
    if dist == 0:
        # Random small push
        angle = random.uniform(0, 2 * math.pi)
        return config.repulsion_max_force * math.cos(angle), config.repulsion_max_force * math.sin(angle)

    if dist >= config.repulsion_distance:
        return 0.0, 0.0

    # Repulsion increases as distance decreases
    force = config.repulsion_max_force * (1 - dist / config.repulsion_distance)
    fx = -force * dx / dist  # Push away
    fy = -force * dy / dist
    return fx, fy


def run_physics_step(words: List[WordState], phrase_groups: List[List[WordState]], config: PhysicsConfig) -> None:
    """Run one step of physics simulation."""
    # Reset velocities for this step
    for word in words:
        word.vx = 0.0
        word.vy = 0.0

    # Apply gravity between all pairs
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            fx, fy = apply_gravity(words[i], words[j], config)
            words[i].apply_force(fx, fy)
            words[j].apply_force(-fx, -fy)

    # Apply phrase impulses
    for group in phrase_groups:
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                fx, fy = apply_phrase_impulse(group[i], group[j], config)
                group[i].apply_force(fx, fy)
                group[j].apply_force(-fx, -fy)

    # Apply repulsion between all pairs
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            fx, fy = apply_repulsion(words[i], words[j], config)
            words[i].apply_force(fx, fy)
            words[j].apply_force(-fx, -fy)

    # Step positions
    for word in words:
        word.step(config.damping, (config.width, config.height))


def run_simulation(words: List[WordState], phrase_groups: List[List[WordState]], config: PhysicsConfig) -> None:
    """Run full physics simulation for configured steps."""
    for _ in range(config.steps):
        run_physics_step(words, phrase_groups, config)
