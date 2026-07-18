"""FastAPI composition root for the V2.7 event graph."""

from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.core.settings import settings
from server.core.database import init_db
from server.core.exceptions import register_exception_handlers
from server.modules.training.api.router import router as training_router
from server.modules.hive.api.router import router as hive_router
from server.modules.hive.api.query_router import router as query_scene_router
from server.modules.universe.api.router import router as universe_router


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
    app.include_router(training_router)
    app.include_router(hive_router)
    app.include_router(query_scene_router)
    app.include_router(universe_router)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "model": "role-free-event-graph",
            "version": "v2.7",
        }

    return app


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Create the fresh role-free schema; no old-scene backfill is run."""
    init_db()
    yield
    # Shutdown: cleanup if needed


app = create_app()
