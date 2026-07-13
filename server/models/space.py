"""Space model - local nebula space."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import json


@dataclass
class Space:
    """Local space inside a host cloud."""
    id: Optional[int] = None
    host_cloud_id: int = 0
    layer_id: int = 0
    mode: str = "structural"  # "structural" or "semantic"
    coordinate_dimensions: int = 2
    scale: float = 1.0
    config_json: str = "{}"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # Computed/transient fields
    _config: Dict[str, Any] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if isinstance(self.config_json, str):
            try:
                self._config = json.loads(self.config_json)
            except json.JSONDecodeError:
                self._config = {}
    
    @property
    def config(self) -> Dict[str, Any]:
        return self._config
    
    @config.setter
    def config(self, value: Dict[str, Any]):
        self._config = value
        self.config_json = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "host_cloud_id": self.host_cloud_id,
            "layer_id": self.layer_id,
            "mode": self.mode,
            "coordinate_dimensions": self.coordinate_dimensions,
            "scale": self.scale,
            "config": self._config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class Layer:
    """Scale layer definition."""
    id: Optional[int] = None
    name: str = ""
    order_index: int = 0
    scale: float = 1.0
    layer_type: str = ""
    config_json: str = "{}"
    created_at: Optional[datetime] = None
    
    # Computed/transient fields
    _config: Dict[str, Any] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if isinstance(self.config_json, str):
            try:
                self._config = json.loads(self.config_json)
            except json.JSONDecodeError:
                self._config = {}
    
    @property
    def config(self) -> Dict[str, Any]:
        return self._config
    
    @config.setter
    def config(self, value: Dict[str, Any]):
        self._config = value
        self.config_json = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "order_index": self.order_index,
            "scale": self.scale,
            "layer_type": self.layer_type,
            "config": self._config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }