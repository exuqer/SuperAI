"""V2.7 graph dialogue request contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HiveCreateRequest(BaseModel):
    max_cells: int = Field(default=24, ge=1, le=128)
    conversation_id: str = Field(default="", max_length=160)


class HiveQueryRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    resolved_mode: str | None = Field(
        default=None,
        pattern="^(NEW_QUERY|FOLLOW_UP|CORRECTION)$",
    )
    retrieval_scope: str = Field(
        default="LOCAL_THEN_GLOBAL",
        pattern="^(LOCAL_ONLY|LOCAL_THEN_GLOBAL|GLOBAL_ONLY)$",
    )


class HiveVibrationRequest(BaseModel):
    steps: int = Field(default=3, ge=1, le=32)
    config: dict[str, Any] = Field(default_factory=dict)
