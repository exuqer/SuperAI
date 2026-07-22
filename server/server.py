"""FastAPI composition root for SuperAI V3.0."""

from __future__ import annotations

from contextlib import asynccontextmanager
import sqlite3
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from server.core.settings import settings
from server.core.database import init_db
from server.core.exceptions import register_exception_handlers
from server.modules.training.api.router import router as training_router
from server.modules.hive.api.router import router as hive_router
from server.modules.hive.api.query_router import router as query_scene_router
from server.modules.universe.api.router import router as universe_router
from server.modules.testing.api.router import router as testing_router
from server.v2.graph_repository import GraphRepository
from server.v2.graph_schema import SCHEMA_VERSION
from server.v2.russian_morphology import RussianMorphology


def _field_revision() -> int:
    try:
        with GraphRepository().transaction() as conn:
            row = conn.execute("SELECT COALESCE(MAX(revision),0) FROM field_revisions").fetchone()
            return int(row[0])
    except (sqlite3.Error, OSError, RuntimeError):
        return 0


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    # Register exception handlers
    register_exception_handlers(app)

    # Include routers
    app.include_router(training_router, prefix=settings.api_prefix)
    app.include_router(hive_router, prefix=settings.api_prefix)
    app.include_router(query_scene_router, prefix=settings.api_prefix)
    app.include_router(testing_router, prefix=settings.api_prefix)
    app.include_router(
        universe_router,
        prefix=settings.api_prefix.rsplit("/v", 1)[0],
    )

    @app.middleware("http")
    async def request_identity(request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.request_id
        return response

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        # Liveness only.  Dependency and pipeline readiness belongs to
        # /api/readiness and must never be guessed here.
        return {
            "status": "alive",
            "model": "superai-spatial-semantic-reasoning",
            "version": settings.app_version,
        }

    @app.get("/api/readiness")
    async def readiness() -> dict[str, object]:
        checks: dict[str, str] = {
            "database": "missing",
            "schema": "missing",
            "morphology": "missing",
            "query_pipeline": "degraded",
            "debug_payload": "ok",
            "indexes": "unknown",
        }
        try:
            repository = GraphRepository()
            with repository.transaction() as conn:
                conn.execute("SELECT 1").fetchone()
                required = {"knowledge_sources", "graph_events", "dialogue_turns", "dialogue_context_states"}
                present = {
                    str(row[0])
                    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                }
                checks["database"] = "ok"
                version = conn.execute("SELECT value FROM graph_meta WHERE key='schema_version'").fetchone()
                checks["schema"] = "ok" if version and str(version[0]) == str(SCHEMA_VERSION) and required <= present else "incompatible"
                checks["indexes"] = "ok" if required <= present else "missing"
                checks["event_graph"] = "ready"
                checks["semantic_field"] = "ready" if "semantic_clouds" in present and "field_revisions" in present else "missing"
                checks["latent_dimensions"] = "ready" if "latent_dimensions" in present else "missing"
                checks["reasoning_pipeline"] = "ready"
        except (sqlite3.Error, OSError, RuntimeError):
            checks["database"] = "unavailable"
        morphology = RussianMorphology()
        checks["morphology"] = "ok" if morphology.available else "missing"
        if checks["database"] == "ok" and checks["schema"] == "ok" and morphology.available:
            checks["query_pipeline"] = "ok"
        ready = all(checks[name] == "ok" for name in ("database", "schema", "morphology", "query_pipeline", "indexes"))
        return {"status": "ready" if ready else "not_ready", "checks": checks, "event_graph": checks.get("event_graph", "missing"), "semantic_field": checks.get("semantic_field", "missing"), "field_revision": _field_revision(), "latent_dimensions": checks.get("latent_dimensions", "missing"), "morphology": checks["morphology"], "reasoning_pipeline": checks.get("reasoning_pipeline", "missing")}

    return app


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Create the fresh role-free schema; no old-scene backfill is run."""
    init_db()
    yield
    # Shutdown: cleanup if needed


app = create_app()
