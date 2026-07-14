"""Pydantic DTOs for model API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class CloudResponse(BaseModel):
    """Cloud response."""
    id: int
    cloud_type: str
    canonical_name: str
    mass: float
    density: float
    stability: float
    base_activation: float
    observation_count: int
    metadata: dict[str, Any]
    created_at: str
    updated_at: str


class SpaceResponse(BaseModel):
    """Space response."""
    id: int
    space_type: str
    owner_cloud_id: int | None = None
    parent_space_id: int | None = None
    dimensionality: int
    random_seed: int
    metadata: dict[str, Any]
    created_at: str


class PlacementResponse(BaseModel):
    """Placement response."""
    id: int
    cloud_id: int
    space_id: int
    x: float
    y: float
    z: float | None = None
    radius: float
    local_activation: float
    local_density: float
    local_gravity: float
    local_stability_modifier: float
    metadata: dict[str, Any]
    created_at: str
    updated_at: str


class StatsResponse(BaseModel):
    """Model statistics response."""
    clouds_total: int
    clouds_by_type: dict[str, int]
    spaces_total: int
    spaces_by_type: dict[str, int]
    placements_total: int
    unique_word_forms: int
    scene_components_total: int
    structural_components_total: int
    concepts_total: int


class NormalizedSpaceResponse(BaseModel):
    """Normalized space response."""
    space: SpaceResponse
    clouds: dict[str, CloudResponse]
    placements: list[PlacementResponse]
    stats: StatsResponse


class StructureResponse(BaseModel):
    """Word structure response."""
    cloud: CloudResponse
    structure_space: SpaceResponse | None = None
    components: list[dict[str, Any]]
    clouds: dict[str, CloudResponse]


class SceneComponentResponse(BaseModel):
    """Scene component response."""
    id: int
    placement_id: int
    cloud_id: int
    lexeme_cloud_id: int | None = None
    token_index: int
    grammatical_role: str
    dependency_role: str | None = None
    head_component_id: int | None = None
    confidence: float
    morphology: dict[str, Any]


class SceneResponse(BaseModel):
    """Scene response."""
    cloud_id: int
    scene_space_id: int
    sentence_text: str
    canonical_text: str
    observation_count: int
    parser_version: str
    components: list[SceneComponentResponse]
    created_at: str
    updated_at: str


class TrainedModelSnapshotResponse(BaseModel):
    """Trained model snapshot response."""
    schema_version: int
    stats: StatsResponse
    model: dict[str, list[dict[str, Any]]]


class PhysicsTickResponse(BaseModel):
    """Physics tick response."""
    space_id: int
    updates: list[dict[str, Any]]


class ClearModelResponse(BaseModel):
    """Clear model response."""
    success: bool
    stats: StatsResponse


class InvariantViolation(BaseModel):
    """Invariant violation."""
    check: str
    violations: list[str]


class InvariantCheckResponse(BaseModel):
    """Invariant check response."""
    passed: bool
    violations: list[str]
    checks: dict[str, InvariantViolation]