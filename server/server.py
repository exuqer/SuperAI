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
from server.modules.hive.api.query_router import router as query_scene_router
from server.modules.hive.api.dynamics_router import router as dynamics_router
from server.v2.taxonomy_api import router as taxonomy_router
from server.v2.knowledge_api import router as knowledge_router


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
    app.include_router(query_scene_router)
    app.include_router(dynamics_router)
    app.include_router(taxonomy_router)
    app.include_router(knowledge_router)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": "cloud-space-placement", "version": "v2"}

    return app


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Application lifespan handler."""
    # Startup schema migration is also the safe place to backfill semantic
    # evidence for scenes saved by an older version.  Query processing stays
    # bounded and never performs a full-model migration.
    init_db()
    from server.v2.repository import V2Repository
    from server.v2.domain_packs import DomainPackService
    from server.v2.event_core import UniversalEventPipeline
    from server.v2.semantic_fog import SemanticFogService
    from server.v2.training import RussianMorphology

    repository = V2Repository()
    with repository.transaction() as conn:
        pipeline = UniversalEventPipeline(repository, RussianMorphology())
        missing_events = conn.execute(
            """SELECT s.cloud_id FROM scenes s
               LEFT JOIN events e ON e.source_scene_id=s.cloud_id
               WHERE e.id IS NULL ORDER BY s.cloud_id"""
        ).fetchall()
        for scene in missing_events:
            pipeline.materialize_scene(conn, int(scene["cloud_id"]))
        global_space = conn.execute(
            "SELECT id FROM spaces WHERE space_type='global_field' LIMIT 1"
        ).fetchone()
        if global_space:
            SemanticFogService(repository).backfill(conn, int(global_space["id"]))
    pack = settings.load_domain_pack
    if not pack and settings.load_demo_knowledge:
        pack = "demo_food"
    if pack:
        DomainPackService(repository).load(pack)
    yield
    # Shutdown: cleanup if needed


# Create the app instance for backward compatibility
app = create_app()
