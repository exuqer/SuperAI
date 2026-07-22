"""Explicit and protected reset API for reproducible experiments."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

from server.core.settings import settings
from server.v2.testing_reset import ResetMode, ResetScope, TestingResetService


router = APIRouter(prefix="/testing", tags=["testing-reset"])


class TestingResetRequest(BaseModel):
    scope: ResetScope = ResetScope.FULL_TEST_STATE
    mode: ResetMode = ResetMode.FRESH_SCHEMA
    confirmation: str | None = None


class TestingRebuildRequest(BaseModel):
    confirmation: str | None = None


def _client_host(request: Request) -> str:
    return str(request.client.host if request.client else "")


def _authorized(request: Request, token: str) -> bool:
    if settings.admin_token and token == settings.admin_token:
        return True
    if _client_host(request) in {"127.0.0.1", "::1", "localhost", "testclient"}:
        return True
    return settings.allow_test_reset and not settings.test_reset_localhost_only


def _guard(request: Request, token: str, confirmation: str | None) -> None:
    if confirmation is not None and confirmation != settings.test_reset_confirmation:
        raise HTTPException(status_code=422, detail="invalid reset confirmation")
    if not _authorized(request, token):
        raise HTTPException(status_code=403, detail="test reset endpoint is disabled")


@router.post("/reset")
async def reset_test_space(
    payload: TestingResetRequest,
    request: Request,
    x_admin_token: str = Header(default=""),
) -> dict[str, Any]:
    _guard(request, x_admin_token, payload.confirmation)
    try:
        return TestingResetService().reset(
            payload.scope,
            payload.mode,
            requested_by=f"http:{_client_host(request)}",
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/rebuild-derived-space")
async def rebuild_derived_space(
    payload: TestingRebuildRequest,
    request: Request,
    x_admin_token: str = Header(default=""),
) -> dict[str, Any]:
    _guard(request, x_admin_token, payload.confirmation)
    return TestingResetService().rebuild_derived_space()
