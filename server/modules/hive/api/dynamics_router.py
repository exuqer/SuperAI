from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from server.modules.hive.application.services import HiveService
from server.modules.hive.api.dto import HiveVibrationRequest
from server.modules.hive.api.router import get_hive_service


router = APIRouter(prefix="/api/hive", tags=["hive-dynamics"])


@router.get("/{hive_id}/dynamics")
async def dynamics(hive_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.dynamics_state(hive_id)


@router.get("/{hive_id}/dynamics/history")
async def dynamics_history(hive_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return {"history": service.dynamics_history(hive_id)}


@router.post("/{hive_id}/dynamics/step")
async def dynamics_step(hive_id: str, request: HiveVibrationRequest, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.dynamics_step(hive_id, request.config)


@router.post("/{hive_id}/dynamics/reset")
async def dynamics_reset(hive_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.dynamics_reset(hive_id)


@router.post("/{hive_id}/dynamics/replay")
async def dynamics_replay(hive_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return {"snapshots": service.dynamics_history(hive_id)}


@router.get("/{hive_id}/dynamics/node/{cell_id}")
async def dynamics_node(hive_id: str, cell_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.dynamics_node(hive_id, cell_id)


@router.get("/{hive_id}/dynamics/evictions")
async def dynamics_evictions(hive_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return {"evictions": service.dynamics_evictions(hive_id)}
