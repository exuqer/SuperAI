"""Local nebula physics simulation with spatial index."""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Set

from server.models.cloud import Cloud, CloudPlacement
from server.repositories.cloud_repository import CloudPlacementRepository
from server.services.spatial_index import SpatialGrid, PhysicsConfig, compute_overlap
from server.services.activation import ActivationManager, spread_activation_in_space


# Backward compatibility
Vector = List[float]


@dataclass
class ConceptState:
    """Legacy concept state for backward compatibility."""
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
    """Legacy physics config for backward compatibility."""
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
    
    # New nebula physics config
    depth: float = 0.0
    grid_cell_size: float = 100.0
    coactivation_attraction_strength: float = 100.0
    base_repulsion_strength: float = 50.0
    max_force: float = 10.0
    max_velocity: float = 50.0
    max_displacement_per_tick: float = 20.0
    stability_damping_factor: float = 0.5
    min_stability: float = 0.0
    max_stability: float = 1.0
    activation_decay: float = 0.95
    activation_spread_factor: float = 0.3
    min_activation: float = 0.01
    overlap_attraction_factor: float = 0.5
    ticks_per_second: int = 20
    max_ticks_per_step: int = 5
    seed: int = 42
    deterministic: bool = True


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


# ============================================================
# NEW: Local Space Physics
# ============================================================

@dataclass
class PlacementState:
    """Runtime state for a cloud placement during physics simulation."""
    placement: CloudPlacement
    cloud: Cloud
    
    @property
    def x(self) -> float:
        return self.placement.x
    
    @x.setter
    def x(self, value: float):
        self.placement.x = value
    
    @property
    def y(self) -> float:
        return self.placement.y
    
    @y.setter
    def y(self, value: float):
        self.placement.y = value
    
    @property
    def z(self) -> float:
        return self.placement.z
    
    @z.setter
    def z(self, value: float):
        self.placement.z = value
    
    @property
    def vx(self) -> float:
        return self.placement.velocity_x
    
    @vx.setter
    def vx(self, value: float):
        self.placement.velocity_x = value
    
    @property
    def vy(self) -> float:
        return self.placement.velocity_y
    
    @vy.setter
    def vy(self, value: float):
        self.placement.velocity_y = value
    
    @property
    def vz(self) -> float:
        return self.placement.velocity_z
    
    @vz.setter
    def vz(self, value: float):
        self.placement.velocity_z = value
    
    @property
    def radius(self) -> float:
        return self.placement.radius
    
    @property
    def mass(self) -> float:
        return self.placement.mass
    
    @property
    def activation(self) -> float:
        return self.placement.activation
    
    @activation.setter
    def activation(self, value: float):
        self.placement.activation = value
    
    @property
    def stability(self) -> float:
        return self.cloud.stability
    
    @property
    def fixed(self) -> bool:
        return self.placement.fixed


class LocalSpacePhysics:
    """Physics simulation for a single local space."""
    
    def __init__(self, space_id: int, config: PhysicsConfig = None):
        self.space_id = space_id
        self.config = config or PhysicsConfig()
        self.grid = SpatialGrid(
            cell_size=self.config.grid_cell_size,
            width=self.config.width,
            height=self.config.height,
            depth=self.config.depth,
        )
        self.placements: Dict[int, PlacementState] = {}  # placement_id -> state
        self.active_placement_ids: Set[int] = set()
        self.random = random.Random(self.config.seed)
        self.tick_count = 0
    
    def add_placement(self, placement: CloudPlacement, cloud: Cloud) -> None:
        """Add a placement to the simulation."""
        state = PlacementState(placement=placement, cloud=cloud)
        self.placements[placement.id] = state
        self.grid.insert(placement.id, placement.x, placement.y, placement.radius, placement.z)
        
        if placement.activation > self.config.min_activation:
            self.active_placement_ids.add(placement.id)
    
    def remove_placement(self, placement_id: int) -> None:
        """Remove a placement from the simulation."""
        if placement_id in self.placements:
            self.grid.remove(placement_id)
            del self.placements[placement_id]
            self.active_placement_ids.discard(placement_id)
    
    def get_active_placements(self) -> List[PlacementState]:
        """Get currently active placements for simulation."""
        return [self.placements[pid] for pid in self.active_placement_ids if pid in self.placements]
    
    def get_all_placements(self) -> List[PlacementState]:
        return list(self.placements.values())
    
    def step(self, activation_manager: ActivationManager = None) -> List[Tuple[int, float, float]]:
        """
        Run one physics tick.
        Returns list of (placement_id, new_x, new_y) for batch update.
        """
        self.tick_count += 1
        
        # Get active placements
        active = self.get_active_placements()
        if not active:
            return []
        
        # 1. Decay activation
        if activation_manager:
            activation_manager.decay_activations([s.placement for s in active])
        
        # 2. Compute forces for each active placement
        forces: Dict[int, Tuple[float, float, float]] = {}
        
        for state in active:
            if state.fixed:
                continue
            
            fx, fy, fz = 0.0, 0.0, 0.0
            
            # Find nearby placements using spatial index
            # Use larger search radius to ensure we find all relevant neighbors
            nearby = self.grid.get_nearby(state.placement.id, max(state.radius * 8, 300.0))
            
            for other_id, distance in nearby:
                other = self.placements.get(other_id)
                if not other:
                    continue
                
                # Co-activation attraction
                coact_force = self._compute_coactivation_force(state, other, distance)
                
                # Overlap-based attraction
                overlap_force = self._compute_overlap_force(state, other, distance)
                
                # Repulsion
                repulsion_force = self._compute_repulsion_force(state, other, distance)
                
                fx += coact_force[0] + overlap_force[0] + repulsion_force[0]
                fy += coact_force[1] + overlap_force[1] + repulsion_force[1]
                fz += coact_force[2] + overlap_force[2] + repulsion_force[2]
            
            # Stability damping: high stability = less movement
            stability_factor = 1.0 - (state.stability * self.config.stability_damping_factor)
            fx *= stability_factor
            fy *= stability_factor
            fz *= stability_factor
            
            # Add small noise for active clouds only
            if state.activation > self.config.min_activation:
                noise_scale = 0.1 * (1.0 - state.stability)
                fx += self.random.uniform(-noise_scale, noise_scale)
                fy += self.random.uniform(-noise_scale, noise_scale)
                if self.config.depth > 0:
                    fz += self.random.uniform(-noise_scale, noise_scale)
            
            forces[state.placement.id] = (fx, fy, fz)
        
        # 3. Apply forces and update positions
        updates = []
        for state in active:
            if state.fixed or state.placement.id not in forces:
                continue
            
            fx, fy, fz = forces[state.placement.id]
            
            # Apply force to velocity
            state.vx = finite(state.vx + fx / state.mass)
            state.vy = finite(state.vy + fy / state.mass)
            state.vz = finite(state.vz + fz / state.mass)
            
            # Damping
            state.vx *= self.config.damping
            state.vy *= self.config.damping
            state.vz *= self.config.damping
            
            # Clamp velocity
            speed = math.sqrt(state.vx**2 + state.vy**2 + state.vz**2)
            if speed > self.config.max_velocity:
                scale = self.config.max_velocity / speed
                state.vx *= scale
                state.vy *= scale
                state.vz *= scale
            
            # Update position
            new_x = state.x + state.vx
            new_y = state.y + state.vy
            new_z = state.z + state.vz
            
            # Clamp displacement per tick
            dx = new_x - state.x
            dy = new_y - state.y
            dz = new_z - state.z
            disp = math.sqrt(dx**2 + dy**2 + dz**2)
            if disp > self.config.max_displacement_per_tick:
                scale = self.config.max_displacement_per_tick / disp
                new_x = state.x + dx * scale
                new_y = state.y + dy * scale
                new_z = state.z + dz * scale
            
            # Boundary constraints
            new_x = max(self.config.boundary_margin, min(self.config.width - self.config.boundary_margin, new_x))
            new_y = max(self.config.boundary_margin, min(self.config.height - self.config.boundary_margin, new_y))
            if self.config.depth > 0:
                new_z = max(self.config.boundary_margin, min(self.config.depth - self.config.boundary_margin, new_z))
            
            # Update grid
            self.grid.update_position(state.placement.id, new_x, new_y, new_z)
            
            updates.append((state.placement.id, new_x, new_y))
        
        # 4. Spread activation
        if activation_manager:
            for state in active:
                if state.activation > self.config.min_activation:
                    spread_activation_in_space(
                        self.grid, self.placements, state.placement.id, state.activation, self.config
                    )
        
        # 5. Return inactive placements toward stable position
        self._restore_inactive_placements()
        
        return updates
    
    def _compute_coactivation_force(self, a: PlacementState, b: PlacementState, 
                                     distance: float) -> Tuple[float, float, float]:
        """Attraction force from co-activation."""
        if a.activation <= 0 or b.activation <= 0:
            return (0.0, 0.0, 0.0)
        
        # Get co-activation strength from database
        # For now, use activation product as proxy
        strength = a.activation * b.activation * self.config.coactivation_attraction_strength
        
        if distance < 1e-6:
            # Random direction
            angle = self.random.random() * 2 * math.pi
            return (strength * math.cos(angle), strength * math.sin(angle), 0.0)
        
        # Maximum force for guaranteed movement
        force = min(strength / max(distance * 0.00001, 1.0), self.config.max_force * 1000)
        dx = b.x - a.x
        dy = b.y - a.y
        dz = b.z - a.z
        return (force * dx / distance, force * dy / distance, force * dz / distance)
    
    def _compute_overlap_force(self, a: PlacementState, b: PlacementState,
                                distance: float) -> Tuple[float, float, float]:
        """Attraction from cloud overlap."""
        overlap = compute_overlap(a.placement, b.placement)
        if overlap <= 0:
            return (0.0, 0.0, 0.0)
        
        strength = overlap * self.config.overlap_attraction_factor * math.sqrt(a.mass * b.mass)
        
        if distance < 1e-6:
            return (0.0, 0.0, 0.0)
        
        force = min(strength, self.config.max_force)
        dx = b.x - a.x
        dy = b.y - a.y
        dz = b.z - a.z
        return (force * dx / distance, force * dy / distance, force * dz / distance)
    
    def _compute_repulsion_force(self, a: PlacementState, b: PlacementState,
                                  distance: float) -> Tuple[float, float, float]:
        """Repulsion to prevent collapse."""
        threshold = a.radius + b.radius
        if distance >= threshold:
            return (0.0, 0.0, 0.0)
        
        if distance < 1e-6:
            angle = self.random.random() * 2 * math.pi
            force = self.config.base_repulsion_strength
            return (force * math.cos(angle), force * math.sin(angle), 0.0)
        
        force = self.config.base_repulsion_strength * (1.0 - distance / threshold)
        force = min(force, self.config.max_force)
        
        dx = b.x - a.x
        dy = b.y - a.y
        dz = b.z - a.z
        return (-force * dx / distance, -force * dy / distance, -force * dz / distance)
    
    def _restore_inactive_placements(self) -> None:
        """Gently pull inactive placements back to their stable positions."""
        # This would require storing a "home" position
        # For now, just apply stronger damping to inactive
        for state in self.placements.values():
            if state.placement.id not in self.active_placement_ids:
                state.vx *= 0.9
                state.vy *= 0.9
                state.vz *= 0.9
    
    def run_ticks(self, ticks: int, activation_manager: ActivationManager = None) -> List[Tuple[int, float, float]]:
        """Run multiple ticks and return all position updates."""
        all_updates = []
        for _ in range(min(ticks, self.config.max_ticks_per_step)):
            updates = self.step(activation_manager)
            all_updates.extend(updates)
            # Apply position updates to database
            if updates:
                self._apply_updates(updates)
        return all_updates
    
    def _apply_updates(self, updates: List[Tuple[int, float, float]]) -> None:
        """Apply position updates to database."""
        from server.repositories.cloud_repository import CloudPlacementRepository
        placement_repo = CloudPlacementRepository()
        placement_repo.update_positions_batch([
            type('obj', (object,), {'id': pid, 'x': x, 'y': y, 'z': 0.0, 'velocity_x': 0.0, 'velocity_y': 0.0, 'velocity_z': 0.0})()
            for pid, x, y in updates
        ])
    
    def get_placements_in_viewport(self, min_x: float, min_y: float, 
                                    max_x: float, max_y: float,
                                    min_density: float = 0.0) -> List[PlacementState]:
        """Get placements visible in viewport."""
        ids = self.grid.query_rect(min_x, min_y, max_x, max_y)
        result = []
        for pid in ids:
            state = self.placements.get(pid)
            if state and state.placement.density >= min_density:
                result.append(state)
        return result
    
    def query_density_at(self, x: float, y: float, z: float = 0.0) -> float:
        """Query total density at a point."""
        from server.services.spatial_index import compute_density_at_point
        placements = [s.placement for s in self.placements.values()]
        return compute_density_at_point(placements, x, y, z)


def create_space_physics(space_id: int, placements: List[CloudPlacement], 
                          clouds: Dict[int, Cloud],
                          config: PhysicsConfig = None) -> LocalSpacePhysics:
    """Factory to create physics simulation for a space."""
    physics = LocalSpacePhysics(space_id, config)
    for placement in placements:
        cloud = clouds.get(placement.cloud_id)
        if cloud:
            physics.add_placement(placement, cloud)
    return physics