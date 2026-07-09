from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    text: str
    session_id: str = "default"
    backpack_limit: int | None = None
    include_graph: bool = False
    include_layers: bool = False
    include_trace: bool = False


class DrillRequest(BaseModel):
    session_id: str = "default"
    node_id: str | None = None
    limit: int | None = None


class TrainRequest(BaseModel):
    text: str | None = None
    dataset_path: str | None = None
    session_id: str = "default"
    epochs: int = 1
    max_pairs: int | None = None


class FeedbackRequest(BaseModel):
    result_id: str
    score: int = Field(ge=0, le=5)
    corrected_response: str | None = None


class NodeDetailResponse(BaseModel):
    node: dict[str, Any]
    neighbors: list[dict[str, Any]]
    examples: list[dict[str, Any]]


def model_payload(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
