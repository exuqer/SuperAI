"""Model HTTP routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from server.modules.model.api.dto import RebuildRequest
from server.modules.model.application.services import ModelService


router = APIRouter(prefix="/api/v2", tags=["model"])


def get_model_service() -> ModelService:
    return ModelService()


@router.get("/field")
async def get_field(service: ModelService = Depends(get_model_service)) -> dict[str, Any]:
    return service.get_field()


@router.get("/stats")
async def get_stats(service: ModelService = Depends(get_model_service)) -> dict[str, Any]:
    return service.get_stats()


@router.delete("/model")
async def clear_model(service: ModelService = Depends(get_model_service)) -> dict[str, Any]:
    return service.clear_model()


@router.get("/model")
async def get_model(service: ModelService = Depends(get_model_service)) -> dict[str, Any]:
    return service.get_trained_model_snapshot()


@router.post("/model/rebuild")
async def rebuild_model(
    body: RebuildRequest,
    service: ModelService = Depends(get_model_service),
) -> dict[str, Any]:
    return service.rebuild_model(body.steps)


@router.get("/clouds/{cloud_id}")
async def get_cloud(
    cloud_id: int, service: ModelService = Depends(get_model_service)
) -> dict[str, Any]:
    return service.get_cloud(cloud_id)


@router.get("/clouds/{cloud_id}/structure")
async def get_structure(
    cloud_id: int, service: ModelService = Depends(get_model_service)
) -> dict[str, Any]:
    return service.get_structure(cloud_id)


@router.get("/placements/{placement_id}")
async def get_placement(
    placement_id: int, service: ModelService = Depends(get_model_service)
) -> dict[str, Any]:
    return service.get_placement(placement_id)


@router.get("/spaces/{space_id}")
async def get_space(
    space_id: int, service: ModelService = Depends(get_model_service)
) -> dict[str, Any]:
    return service.get_space(space_id)


@router.post("/spaces/{space_id}/physics/tick")
async def physics_tick(
    space_id: int, service: ModelService = Depends(get_model_service)
) -> dict[str, Any]:
    return service.physics_tick(space_id)


@router.get("/scenes/{scene_id}")
async def get_scene(
    scene_id: int, service: ModelService = Depends(get_model_service)
) -> dict[str, Any]:
    return service.get_scene(scene_id)


@router.get("/debug/invariants")
async def debug_invariants(
    service: ModelService = Depends(get_model_service),
) -> dict[str, Any]:
    return service.debug_invariants()
