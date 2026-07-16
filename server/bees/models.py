from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping
from uuid import uuid4

from server.spaces import SpaceLevel


@dataclass(frozen=True)
class BeeTask:
    target_space: SpaceLevel | str
    task: str
    desired_features: dict[str, Any]
    budget: int = 20
    min_relevance: float = 0.65
    source_fragment: str = ""
    max_candidates: int = 8
    max_downward_depth: int = 2
    task_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if self.budget < 0:
            raise ValueError("budget must be non-negative")
        if not 0 <= self.min_relevance <= 1:
            raise ValueError("min_relevance must be between 0 and 1")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")

    @property
    def space_name(self) -> str:
        return (
            self.target_space.value
            if isinstance(self.target_space, SpaceLevel)
            else self.target_space
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["target_space"] = self.space_name
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BeeTask":
        return cls(**dict(value))


@dataclass(frozen=True)
class NectarComponent:
    dimension: str
    value: Any
    activation: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NectarPacket:
    origin_space: str
    source_id: str
    components: tuple[NectarComponent, ...]
    confidence: float
    utility: float
    cost: int
    provenance: dict[str, Any]
    bee_type: str
    task_id: str
    packet_id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["components"] = [item.to_dict() for item in self.components]
        return value
