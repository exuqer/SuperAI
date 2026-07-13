"""Spatial index for efficient nebula physics simulation."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict


@dataclass
class SpatialGrid:
    """Uniform grid spatial index for 2D/3D space."""
    cell_size: float
    width: float
    height: float
    depth: float = 0.0
    
    # Grid cells: (gx, gy, gz) -> set of placement_ids
    _cells: Dict[Tuple[int, int, int], Set[int]] = None
    
    # Placement positions cache: placement_id -> (x, y, z, radius)
    _placements: Dict[int, Tuple[float, float, float, float]] = None
    
    def __post_init__(self):
        self._cells = defaultdict(set)
        self._placements = {}
    
    def _get_cell_coords(self, x: float, y: float, z: float = 0.0) -> Tuple[int, int, int]:
        gx = int(x // self.cell_size)
        gy = int(y // self.cell_size)
        gz = int(z // self.cell_size) if self.depth > 0 else 0
        return (gx, gy, gz)
    
    def _get_affected_cells(self, x: float, y: float, radius: float, z: float = 0.0) -> List[Tuple[int, int, int]]:
        """Get all grid cells that intersect with a circle/sphere."""
        min_x = x - radius
        max_x = x + radius
        min_y = y - radius
        max_y = y + radius
        min_z = z - radius if self.depth > 0 else 0
        max_z = z + radius if self.depth > 0 else 0
        
        min_gx = int(min_x // self.cell_size)
        max_gx = int(max_x // self.cell_size)
        min_gy = int(min_y // self.cell_size)
        max_gy = int(max_y // self.cell_size)
        min_gz = int(min_z // self.cell_size) if self.depth > 0 else 0
        max_gz = int(max_z // self.cell_size) if self.depth > 0 else 0
        
        cells = []
        for gx in range(min_gx, max_gx + 1):
            for gy in range(min_gy, max_gy + 1):
                for gz in range(min_gz, max_gz + 1):
                    cells.append((gx, gy, gz))
        return cells
    
    def insert(self, placement_id: int, x: float, y: float, radius: float, z: float = 0.0) -> None:
        """Insert or update a placement in the spatial index."""
        # Remove old position if exists
        self.remove(placement_id)
        
        self._placements[placement_id] = (x, y, z, radius)
        cells = self._get_affected_cells(x, y, radius, z)
        for cell in cells:
            self._cells[cell].add(placement_id)
    
    def remove(self, placement_id: int) -> None:
        """Remove a placement from the spatial index."""
        if placement_id not in self._placements:
            return
        x, y, z, radius = self._placements[placement_id]
        cells = self._get_affected_cells(x, y, radius, z)
        for cell in cells:
            self._cells[cell].discard(placement_id)
            if not self._cells[cell]:
                del self._cells[cell]
        del self._placements[placement_id]
    
    def update_position(self, placement_id: int, x: float, y: float, z: float = 0.0) -> None:
        """Update position (radius unchanged)."""
        if placement_id not in self._placements:
            return
        old_x, old_y, old_z, radius = self._placements[placement_id]
        # Remove from old cells
        old_cells = self._get_affected_cells(old_x, old_y, radius, old_z)
        for cell in old_cells:
            self._cells[cell].discard(placement_id)
            if not self._cells[cell]:
                del self._cells[cell]
        # Add to new cells
        self._placements[placement_id] = (x, y, z, radius)
        new_cells = self._get_affected_cells(x, y, radius, z)
        for cell in new_cells:
            self._cells[cell].add(placement_id)
    
    def query_radius(self, x: float, y: float, radius: float, z: float = 0.0) -> List[int]:
        """Find all placements within radius of a point."""
        cells = self._get_affected_cells(x, y, radius, z)
        result = set()
        for cell in cells:
            result.update(self._cells.get(cell, set()))
        
        # Filter by actual distance
        filtered = []
        for pid in result:
            px, py, pz, pr = self._placements.get(pid, (0, 0, 0, 0))
            dx = px - x
            dy = py - y
            dz = pz - z if self.depth > 0 else 0
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist <= radius + pr:
                filtered.append(pid)
        return filtered
    
    def query_rect(self, min_x: float, min_y: float, max_x: float, max_y: float) -> List[int]:
        """Find all placements intersecting a rectangle."""
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        radius = max(max_x - min_x, max_y - min_y) / 2 + 100  # generous radius
        return self.query_radius(cx, cy, radius)
    
    def get_nearby(self, placement_id: int, max_distance: float) -> List[Tuple[int, float]]:
        """Get nearby placements with distances."""
        if placement_id not in self._placements:
            return []
        x, y, z, radius = self._placements[placement_id]
        candidates = self.query_radius(x, y, max_distance + radius, z)
        result = []
        for pid in candidates:
            if pid == placement_id:
                continue
            px, py, pz, pr = self._placements[pid]
            dx = px - x
            dy = py - y
            dz = pz - z if self.depth > 0 else 0
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist <= max_distance + radius + pr:
                result.append((pid, dist))
        return result
    
    def get_all_in_cell(self, x: float, y: float, z: float = 0.0) -> List[int]:
        """Get all placements in the same cell as a point."""
        cell = self._get_cell_coords(x, y, z)
        return list(self._cells.get(cell, set()))
    
    def clear(self) -> None:
        self._cells.clear()
        self._placements.clear()
    
    @property
    def placement_count(self) -> int:
        return len(self._placements)
    
    @property
    def cell_count(self) -> int:
        return len(self._cells)


def compute_overlap(placement_a, placement_b) -> float:
    """Compute overlap ratio between two cloud placements (0.0 to 1.0)."""
    dx = placement_a.x - placement_b.x
    dy = placement_a.y - placement_b.y
    dz = placement_a.z - placement_b.z
    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    radius_a = placement_a.radius
    radius_b = placement_b.radius
    
    if distance >= radius_a + radius_b:
        return 0.0
    
    # Approximate overlap
    min_radius = min(radius_a, radius_b)
    if min_radius <= 0:
        return 0.0
    
    overlap = (radius_a + radius_b - distance) / min_radius
    return max(0.0, min(1.0, overlap))


def compute_density_at_point(placements: List, x: float, y: float, z: float = 0.0) -> float:
    """Compute total density at a point from all placements."""
    total = 0.0
    for p in placements:
        dx = p.x - x
        dy = p.y - y
        dz = p.z - z
        dist_sq = dx*dx + dy*dy + dz*dz
        sigma = p.radius / 2.0  # sigma = radius/2 for Gaussian
        if sigma <= 0:
            continue
        # Gaussian density
        total += p.mass * math.exp(-dist_sq / (2 * sigma * sigma))
    return total


class PhysicsConfig:
    """Configuration for nebula physics simulation."""
    
    def __init__(self):
        # Space bounds
        self.width = 1600.0
        self.height = 1000.0
        self.depth = 0.0  # 2D by default
        
        # Grid cell size (should be ~average cloud radius)
        self.grid_cell_size = 100.0
        
        # Forces
        self.coactivation_attraction_strength = 100.0
        self.base_repulsion_strength = 50.0
        self.max_force = 10.0
        
        # Dynamics
        self.damping = 0.85
        self.max_velocity = 50.0
        self.max_displacement_per_tick = 20.0
        
        # Stability
        self.stability_damping_factor = 0.5  # high stability = less movement
        self.min_stability = 0.0
        self.max_stability = 1.0
        
        # Activation
        self.activation_decay = 0.95
        self.activation_spread_factor = 0.3
        self.min_activation = 0.01
        
        # Overlap
        self.overlap_attraction_factor = 0.5
        
        # Simulation
        self.ticks_per_second = 20
        self.max_ticks_per_step = 5
        
        # Determinism
        self.seed = 42
        self.deterministic = True


# Global config instance
physics_config = PhysicsConfig()