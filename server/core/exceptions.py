"""Application exceptions and error handling."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", "")
    return str(value or uuid.uuid4().hex)


def _payload(request: Request, code: str, message: Any, detail: Any = None) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": str(message),
            "detail": detail or {},
            "request_id": _request_id(request),
        }
    }


class AppException(Exception):
    """Base application exception."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail or {}


class NotFoundError(AppException):
    """Resource not found."""

    def __init__(self, resource: str, identifier: str | int) -> None:
        super().__init__(
            f"{resource} not found",
            status_code=404,
            detail={"resource": resource, "id": str(identifier)},
        )


class ValidationError(AppException):
    """Validation error."""

    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message, status_code=422, detail=detail)


class ConflictError(AppException):
    """Conflict error."""

    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message, status_code=409, detail=detail)


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle application exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=_payload(request, exc.__class__.__name__.upper(), exc.message, exc.detail),
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=_payload(request, "HTTP_ERROR", exc.detail),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.exception("unhandled request error request_id=%s route=%s exception=%s", _request_id(request), request.url.path, type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content=_payload(request, "INTERNAL_PIPELINE_ERROR", "Internal server error"),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register exception handlers on FastAPI app."""
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
