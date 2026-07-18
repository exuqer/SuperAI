"""Focused QueryGraph inspection endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from server.modules.hive.application.services import HiveService


router = APIRouter(prefix="/api/v2/graphs", tags=["query-graphs"])


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


def service() -> HiveService:
    return HiveService()


@router.post("/parse")
async def parse_graph(
    request: TextRequest,
    graph_service: HiveService = Depends(service),
) -> dict[str, Any]:
    return graph_service.parse_query(request.text)


@router.get("/hives/{hive_id}")
async def graph_state(
    hive_id: str,
    graph_service: HiveService = Depends(service),
) -> dict[str, Any]:
    return graph_service.query_working_state(hive_id)
