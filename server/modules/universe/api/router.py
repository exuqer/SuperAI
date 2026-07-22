"""Read and visualization API for dynamic micro-universes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from server.core.settings import settings
from server.v2.graph_repository import GraphRepository
from server.v2.semantic_field import SemanticFieldService
from server.v2.universe import UniverseService


router = APIRouter(tags=["universes"])


def semantic_field() -> SemanticFieldService:
    return SemanticFieldService(GraphRepository())


@router.get("/v2/semantic-field")
async def semantic_field_view(
    limit: int = Query(default=200, ge=1, le=2000),
    field: SemanticFieldService = Depends(semantic_field),
) -> dict[str, Any]:
    return field.snapshot(limit=limit)


@router.post("/v2/semantic-field/rollback/{field_revision}")
async def rollback_semantic_field(field_revision: int, field: SemanticFieldService = Depends(semantic_field)) -> dict[str, Any]:
    try:
        return field.restore_revision(field_revision)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"field revision not found: {error.args[0]}") from error


def service() -> UniverseService:
    return UniverseService()


def _not_found(error: KeyError) -> HTTPException:
    return HTTPException(status_code=404, detail=f"not found: {error.args[0]}")


@router.get("/universes")
async def universes(current: UniverseService = Depends(service)) -> dict[str, Any]:
    return current.list_universes()


@router.post("/reset")
async def reset_memory(
    x_admin_token: str = Header(default=""),
    current: UniverseService = Depends(service),
) -> dict[str, Any]:
    """Destructively reset all graph and universe data in the current database."""
    if not settings.admin_token or x_admin_token != settings.admin_token:
        raise HTTPException(status_code=403, detail="destructive endpoint is disabled")
    return current.reset()


@router.get("/export/memory")
async def export_memory(current: UniverseService = Depends(service)) -> dict[str, Any]:
    return current.export_memory()


@router.get("/universes/{universe_id}/base-space")
async def base_space(
    universe_id: str,
    limit: int = Query(default=200, ge=1, le=2000),
    min_mass: float = Query(default=0.0, ge=0.0),
    min_stability: float = Query(default=0.0, ge=0.0, le=1.0),
    selected_context: str = "",
    current: UniverseService = Depends(service),
) -> dict[str, Any]:
    try:
        return current.base_space(universe_id, limit=limit, min_mass=min_mass,
                                  min_stability=min_stability,
                                  selected_context=selected_context)
    except KeyError as error:
        raise _not_found(error) from error


@router.get("/universes/{universe_id}/dimensions")
async def dimensions(
    universe_id: str,
    status: str = "",
    scope: str = "",
    min_stability: float = Query(default=0.0, ge=0.0, le=1.0),
    min_utility: float = Query(default=0.0, ge=0.0, le=1.0),
    owner_cloud_id: str = "",
    current: UniverseService = Depends(service),
) -> dict[str, Any]:
    try:
        return current.dimensions(universe_id, status=status, scope=scope,
                                  min_stability=min_stability,
                                  min_utility=min_utility,
                                  owner_cloud_id=owner_cloud_id)
    except KeyError as error:
        raise _not_found(error) from error


@router.get("/dimensions/{dimension_id}")
async def dimension(
    dimension_id: str,
    current: UniverseService = Depends(service),
) -> dict[str, Any]:
    try:
        return current.dimension(dimension_id)
    except KeyError as error:
        raise _not_found(error) from error


@router.get("/dimensions/{dimension_id}/projections")
async def projections(
    dimension_id: str,
    source_type: str = "",
    limit: int = Query(default=100, ge=1, le=2000),
    min_membership: float = Query(default=0.0, ge=0.0, le=1.0),
    context_id: str = "",
    sort: str = "membership",
    current: UniverseService = Depends(service),
) -> dict[str, Any]:
    try:
        return current.projections(dimension_id, source_type=source_type, limit=limit,
                                   min_membership=min_membership,
                                   context_id=context_id, sort=sort)
    except KeyError as error:
        raise _not_found(error) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/entities/{entity_id}/dimension-profile")
async def entity_profile(
    entity_id: str,
    current: UniverseService = Depends(service),
) -> dict[str, Any]:
    try:
        return current.profile(entity_id)
    except KeyError as error:
        raise _not_found(error) from error


class EntityComparisonRequest(BaseModel):
    entity_ids: list[str] = Field(min_length=2, max_length=2)
    universe_id: str = Field(min_length=1)


@router.post("/entities/compare")
async def compare_entities(
    request: EntityComparisonRequest,
    current: UniverseService = Depends(service),
) -> dict[str, Any]:
    try:
        return current.compare(request.entity_ids, request.universe_id)
    except KeyError as error:
        raise _not_found(error) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


class ProjectionRequest(BaseModel):
    universe_id: str = Field(min_length=1)
    space_type: str = "base"
    dimension_ids: list[str] = Field(default_factory=list, max_length=8)
    projection_method: str = "selected_dimensions"
    limit: int = Field(default=200, ge=1, le=2000)
    filters: dict[str, Any] = Field(default_factory=dict)


@router.post("/visualization/project")
async def visualization_projection(
    request: ProjectionRequest,
    current: UniverseService = Depends(service),
) -> dict[str, Any]:
    try:
        return current.project(request.universe_id, request.model_dump())
    except KeyError as error:
        raise _not_found(error) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/universes/{universe_id}/transitions")
async def transitions(
    universe_id: str,
    entity_id: str = "",
    limit: int = Query(default=200, ge=1, le=2000),
    current: UniverseService = Depends(service),
) -> dict[str, Any]:
    return current.transitions(universe_id, entity_id, limit)


@router.get("/training/history")
async def training_history(
    universe_id: str = "",
    dimension_id: str = "",
    limit: int = Query(default=200, ge=1, le=2000),
    current: UniverseService = Depends(service),
) -> dict[str, Any]:
    return current.history(universe_id, dimension_id, limit)


class AliasRequest(BaseModel):
    alias: str = Field(min_length=1, max_length=160)


@router.put("/dimensions/{dimension_id}/alias")
async def dimension_alias(
    dimension_id: str,
    request: AliasRequest,
    current: UniverseService = Depends(service),
) -> dict[str, Any]:
    try:
        return current.set_alias(dimension_id, request.alias)
    except KeyError as error:
        raise _not_found(error) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
