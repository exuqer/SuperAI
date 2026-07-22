"""V3.0 training request contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000)
    source_type: str = "training"
    independent_key: str = ""
    domain_key: str = ""


class StageRequest(TrainRequest):
    pass


class SourceReferenceRequest(BaseModel):
    source_id: str = Field(..., min_length=1)


class CommitRequest(SourceReferenceRequest):
    manual_validation: bool = False


class RetractionRequest(SourceReferenceRequest):
    reason: str = ""


class ReprocessRequest(SourceReferenceRequest):
    pass


class BatchPreviewRequest(BaseModel):
    sources: list[dict] = Field(..., min_length=1)
    config: dict = Field(default_factory=dict)


class BatchReferenceRequest(BaseModel):
    batch_id: str = Field(..., min_length=1)
