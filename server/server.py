"""FastAPI application for the canonical V2 model - Composition Root."""

from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.core.settings import settings
from server.core.database import init_db
from server.core.exceptions import register_exception_handlers
from server.modules.model.api.router import router as model_router
from server.modules.training.api.router import router as training_router
from server.modules.hive.api.router import router as hive_router


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
    app.include_router(model_router)
    app.include_router(training_router)
    app.include_router(hive_router)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": "cloud-space-placement", "version": "v2"}

    return app


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Application lifespan handler."""
    # Startup: initialize database
    init_db()
    yield
    # Shutdown: cleanup if needed


# Create the app instance for backward compatibility
app = create_app()
