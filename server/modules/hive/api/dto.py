"""Hive API DTOs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HiveCreateRequest(BaseModel):
    """Hive creation request."""
    max_cells: int = Field(default=24, ge=1, le=128)
    conversation_id: str = Field(default="", max_length=160)


class HiveCreateResponse(BaseModel):
    """Hive creation response."""
    hive: dict[str, Any]
    cells: list[dict[str, Any]]
    messages: list[dict[str, Any]]


class HiveQueryRequest(BaseModel):
    """Hive query request."""
    text: str = Field(..., min_length=1)
    resolved_mode: str | None = Field(default=None, pattern="^(NEW_QUERY|LOCAL_RESONANCE|FOLLOW_UP|CORRECTION)$")
    resonance_scope: str = Field(default="LOCAL_THEN_GLOBAL", pattern="^(LOCAL_ONLY|LOCAL_THEN_GLOBAL|GLOBAL_ONLY)$")


class HiveVibrationRequest(BaseModel):
    steps: int = Field(default=3, ge=1, le=32)
    config: dict[str, Any] = Field(default_factory=dict)


class ResonanceRequest(BaseModel):
    input: str = Field(..., min_length=1)
    temperature: float = Field(default=.25, ge=0, le=1)
    max_ticks: int = Field(default=8, ge=1, le=64)
    use_global_memory: bool = True
    save_snapshots: bool = True
    config: dict[str, Any] = Field(default_factory=dict)


class LexicalCandidatesRequest(BaseModel):
    text: str = Field(..., min_length=1)
    use_global_memory: bool = True


class ResonanceImportRequest(BaseModel):
    match_id: str = Field(..., min_length=1)
    include_scenes: bool = False


class ResonanceConceptImportRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    concept_id: str = Field(..., min_length=1)


class HiveExpandRequest(BaseModel):
    target_level: str = Field(default="word_form")
    reason: str = Field(default="manual", max_length=240)
    max_candidates: int = Field(default=5, ge=1, le=32)


class HiveGenerateRequest(BaseModel):
    sentence_plan: dict[str, Any]


class HiveValidateSurfaceRequest(BaseModel):
    surface: str = Field(..., min_length=1)


class HiveComposeFormRequest(BaseModel):
    concept: str = Field(..., min_length=1, max_length=160)
    root: str | None = Field(default=None, max_length=160)
    features: dict[str, Any] = Field(default_factory=dict)


class HiveQueryResponse(BaseModel):
    """Hive query response."""
    hive: dict[str, Any]
    cells: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    message_id: str
    decision: dict[str, Any]
    resonance_events: list[dict[str, Any]]
    external_search: dict[str, Any]
    merge_results: list[dict[str, Any]]
    metrics: dict[str, Any]


class HivePreviewResponse(BaseModel):
    """Hive preview response."""
    decision: str
    external_search_required: bool
    matches: list[dict[str, Any]]
    unresolved_components: list[dict[str, Any]]
    local_anchors: list[dict[str, Any]]
    external_request: dict[str, Any] | None
    reasons: list[str]
    parsed_message: dict[str, Any]


class HiveReasoningRequest(BaseModel):
    """Hive reasoning request."""
    text: str = Field(default="", max_length=10000)
    config: dict[str, Any] = Field(default_factory=dict)


class HiveReasoningResponse(BaseModel):
    """Hive reasoning response."""
    run: dict[str, Any]
    completed_steps: int
    stop_reason: str
    final_state: dict[str, Any]
    hive: dict[str, Any]


class HiveExportRequest(BaseModel):
    """Hive export request."""
    mode: str = Field(default="current")
    run_id: str = Field(default="")
    step: int | None = None
    detail: str = Field(default="full")


class HiveAnalyticsResponse(BaseModel):
    """Read-only experiment analytics for one hive."""
    hive_id: str
    current: dict[str, Any]
    runs: list[dict[str, Any]]
    primary: dict[str, Any] | None
    comparison: dict[str, Any] | None


class HiveRestoreRequest(BaseModel):
    """Hive restore request."""
    run_id: str
    step: int


class HiveDecisionResponse(BaseModel):
    """Hive decision response."""
    id: str
    hive_id: str
    message_id: str
    decision: str
    external_search_required: int
    anchors_json: str
    unresolved_json: str
    reasons_json: str
    metrics_json: str
    created_at: str


class HiveResonanceEventResponse(BaseModel):
    """Hive resonance event response."""
    id: str
    hive_id: str
    message_id: str
    cell_id: str
    component_cloud_id: int | None
    reason: str
    payload_json: str
    created_at: str


class HiveMatchResponse(BaseModel):
    """Hive cell match response."""
    id: int
    decision_id: str
    cell_id: str
    component_id: str
    match_type: str
    local_support: float
    metadata_json: str


class HiveReasoningRunResponse(BaseModel):
    """Hive reasoning run response."""
    id: str
    hive_id: str
    status: str
    reasoning_steps: int
    completed_steps: int
    query_json: str
    config_json: str
    random_seed: int
    stop_reason: str | None
    initial_state_hash: str | None
    final_state_hash: str | None
    created_at: str
    completed_at: str | None


class HiveReasoningSnapshotResponse(BaseModel):
    """Hive reasoning snapshot response."""
    id: str
    run_id: str
    hive_id: str
    step: int
    phase: str
    state_hash: str
    state_json: str
    delta_json: str
    clusters_json: str
    events_json: str
    created_at: str


class HiveDiffResponse(BaseModel):
    """Hive diff response."""
    run_id: str
    from_step: int
    to_step: int
    added_nodes: list[int]
    removed_nodes: list[int]
    changed_nodes: list[int]
    clusters_delta: int
