"""Optional knowledge-pack API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .domain_packs import DomainPackService


router = APIRouter(prefix="/api/v2/knowledge", tags=["knowledge"])


@router.get("/domain-packs")
async def domain_packs() -> dict[str, list[str]]:
    return {"domain_packs": DomainPackService.available()}


@router.post("/domain-packs/{name}/load")
async def load_domain_pack(name: str) -> dict:
    try:
        return DomainPackService().load(name)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="domain pack not found") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
