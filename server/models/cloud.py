"""Cloud model - global nebula entity."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
import json


@dataclass
class Cloud:
    """Global nebula entity - exists independently of any space."""
    id: Optional[int] = None
    layer_id: int = 0
    cloud_type: str = ""
    canonical_name: str = ""
    mass: float = 1.0
    density: float = 1.0
    radius: float = 0.0
    stability: float = 0.0
    activation: float = 0.0
    observation_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_activated_at: Optional[datetime] = None
    metadata_json: str = "{}"
    
    # Computed/transient fields
    _metadata: Dict[str, Any] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        if self.radius == 0.0:
            self.radius = self._compute_radius()
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if isinstance(self.metadata_json, str):
            try:
                self._metadata = json.loads(self.metadata_json)
            except json.JSONDecodeError:
                self._metadata = {}
    
    def _compute_radius(self) -> float:
        """Compute radius from mass and density."""
        import math
        return min(250.0, 22.0 + 12.0 * math.sqrt(max(0.001, self.mass * self.density)))
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata
    
    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        self._metadata = value
        self.metadata_json = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "layer_id": self.layer_id,
            "cloud_type": self.cloud_type,
            "canonical_name": self.canonical_name,
            "mass": self.mass,
            "density": self.density,
            "radius": self.radius,
            "stability": self.stability,
            "activation": self.activation,
            "observation_count": self.observation_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_activated_at": self.last_activated_at.isoformat() if self.last_activated_at else None,
            "metadata": self._metadata,
        }


@dataclass
class CloudPlacement:
    """Local appearance of a cloud in a specific space."""
    id: Optional[int] = None
    space_id: int = 0
    cloud_id: int = 0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    radius: float = 0.0
    density: float = 1.0
    mass: float = 1.0
    activation: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0
    fixed: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "space_id": self.space_id,
            "cloud_id": self.cloud_id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "radius": self.radius,
            "density": self.density,
            "mass": self.mass,
            "activation": self.activation,
            "velocity_x": self.velocity_x,
            "velocity_y": self.velocity_y,
            "velocity_z": self.velocity_z,
            "fixed": self.fixed,
        }


@dataclass
class StructuralComponent:
    """Internal structural composition of a cloud (technical, not semantic)."""
    id: Optional[int] = None
    parent_cloud_id: int = 0
    child_cloud_id: int = 0
    child_placement_id: Optional[int] = None
    position_index: int = 0
    phase: float = 0.0
    weight: float = 1.0
    role: str = ""
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parent_cloud_id": self.parent_cloud_id,
            "child_cloud_id": self.child_cloud_id,
            "child_placement_id": self.child_placement_id,
            "position_index": self.position_index,
            "phase": self.phase,
            "weight": self.weight,
            "role": self.role,
        }