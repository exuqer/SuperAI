"""Training API DTOs."""

from __future__ import annotations


from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    """Training request."""
    text: str = Field(..., min_length=1)


class TrainResponse(BaseModel):
    """Training response."""
    training_run_id: str
    success: bool
    created_clouds: int = 0
    strengthened_clouds: int = 0
    created_spaces: int = 0
    created_placements: int = 0
    created_structures: int = 0
    reused_scenes: int = 0


class TrainingRunResponse(BaseModel):
    """Training run response."""
    id: str
    source_text: str
    source_type: str
    success: int
    created_at: str
    completed_at: str | None = None


class TrainingEventResponse(BaseModel):
    """Training event response."""
    id: int
    training_run_id: str
    event_type: str
    entity_type: str
    entity_id: str
    value_before_json: str | None = None
    value_after_json: str | None = None
    reason: str
    created_at: str


class TrainingObservationResponse(BaseModel):
    """Training observation response."""
    id: int
    training_run_id: str
    source_text: str
    normalized_text: str
    scene_cloud_id: int | None = None
    source_type: str
    created_at: str
    metadata_json: str