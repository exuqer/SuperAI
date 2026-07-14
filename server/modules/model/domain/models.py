"""Domain models for the model module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Cloud:
    """Cloud domain model."""
    id: int
    cloud_type: str
    canonical_name: str
    mass: float = 1.0
    density: float = 1.0
    stability: float = 0.0
    base_activation: float = 0.0
    observation_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    @property
    def is_concept(self) -> bool:
        return self.cloud_type == "concept"

    @property
    def is_scene(self) -> bool:
        return self.cloud_type == "scene"

    @property
    def is_word_form(self) -> bool:
        return self.cloud_type == "word_form"

    @property
    def is_lexeme(self) -> bool:
        return self.cloud_type == "lexeme"

    @property
    def is_character(self) -> bool:
        return self.cloud_type == "character"


@dataclass
class Space:
    """Space domain model."""
    id: int
    space_type: str
    owner_cloud_id: int | None = None
    parent_space_id: int | None = None
    dimensionality: int = 2
    random_seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    @property
    def is_global_field(self) -> bool:
        return self.space_type == "global_field"

    @property
    def is_hive_space(self) -> bool:
        return self.space_type == "hive_space"

    @property
    def is_scene_space(self) -> bool:
        return self.space_type == "scene_space"


@dataclass
class Placement:
    """Cloud placement domain model."""
    id: int
    cloud_id: int
    space_id: int
    x: float
    y: float
    z: float | None = None
    radius: float = 12.0
    local_activation: float = 0.0
    local_density: float = 1.0
    local_gravity: float = 0.0
    local_stability_modifier: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    @property
    def is_global(self) -> bool:
        return self.metadata.get("placement_kind") == "global"


@dataclass
class StructuralComponent:
    """Structural component domain model."""
    id: int
    parent_cloud_id: int
    child_cloud_id: int
    component_index: int
    component_role: str = "unknown"
    weight: float = 1.0
    local_x: float = 0.0
    local_y: float = 0.0
    local_z: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneComponent:
    """Scene component domain model."""
    id: int
    scene_cloud_id: int
    word_form_cloud_id: int
    lexeme_cloud_id: int | None
    placement_id: int
    token_index: int
    grammatical_role: str
    dependency_role: str | None = None
    head_component_id: int | None = None
    confidence: float = 1.0
    morphology: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scene:
    """Scene domain model."""
    cloud_id: int
    scene_space_id: int
    sentence_text: str
    canonical_text: str
    fingerprint: str
    parser_version: str
    observation_count: int = 1
    components: list[SceneComponent] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class ModelStats:
    """Model statistics."""
    clouds_total: int = 0
    clouds_by_type: dict[str, int] = field(default_factory=dict)
    spaces_total: int = 0
    spaces_by_type: dict[str, int] = field(default_factory=dict)
    placements_total: int = 0
    unique_word_forms: int = 0
    structural_components_total: int = 0
    concepts_total: int = 0


@dataclass
class NormalizedSpace:
    """Normalized space with all related data."""
    space: Space
    clouds: dict[str, Cloud]
    placements: list[Placement]
    stats: ModelStats


@dataclass
class Structure:
    """Word structure."""
    cloud: Cloud
    structure_space: Space | None
    components: list[StructuralComponent]
    clouds: dict[str, Cloud]


@dataclass
class TrainedModelSnapshot:
    """Complete trained model snapshot."""
    schema_version: int
    stats: ModelStats
    model: dict[str, list[dict[str, Any]]]