"""HTTP routes for V2.7 role-free graph dialogue."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from server.modules.hive.api.dto import (
    HiveCreateRequest,
    HiveQueryRequest,
    HiveVibrationRequest,
)
from server.modules.hive.application.services import HiveService


router = APIRouter(prefix="/api/v2/hives", tags=["event-graphs"])


def get_hive_service() -> HiveService:
    return HiveService()


@router.post("")
async def create_hive(
    request: HiveCreateRequest,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, Any]:
    return service.create(request.max_cells, request.conversation_id)


@router.post("/query/parse")
async def parse_query_graph(
    request: HiveQueryRequest,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, Any]:
    return service.parse_query(request.text)


@router.get("/{hive_id}")
async def get_hive(
    hive_id: str,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, Any]:
    return service.get_hive(hive_id)


@router.post("/{hive_id}/query")
async def query_hive(
    hive_id: str,
    request: HiveQueryRequest,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, Any]:
    return service.query(
        hive_id,
        request.text,
        request.resolved_mode,
        request.retrieval_scope,
    )


@router.post("/{hive_id}/query/preview")
async def preview_query_graph(
    hive_id: str,
    request: HiveQueryRequest,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, Any]:
    return service.preview(hive_id, request.text)


@router.get("/{hive_id}/trace")
async def get_trace(
    hive_id: str,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, Any]:
    state = service.query_working_state(hive_id)
    return {"trace": state.get("trace") or {}}


@router.get("/{hive_id}/bindings")
async def get_bindings(
    hive_id: str,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, Any]:
    state = service.query_working_state(hive_id)
    return {
        "candidate_bindings": state.get("candidate_bindings") or [],
        "selected_binding": state.get("selected_binding"),
        "rejected_events": state.get("rejected_events") or [],
    }


@router.get("/{hive_id}/space-export")
async def export_chat_space(
    hive_id: str,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, Any]:
    """Portable internal state for the selected dialogue workspace."""
    return service.export(hive_id, mode="chat-space", detail="full")


@router.post("/{hive_id}/rank/step")
async def rank_step(
    hive_id: str,
    request: HiveVibrationRequest,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, Any]:
    return service.vibration_step(hive_id, request.config)


@router.post("/{hive_id}/rank/run")
async def rank_run(
    hive_id: str,
    request: HiveVibrationRequest,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, Any]:
    return service.vibration_run(hive_id, request.steps, request.config)
