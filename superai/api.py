"""FastAPI transport adapter. Routes validate and delegate; no domain logic lives here."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Query, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from .contracts import AccessScope, ErrorEnvelope, GenomeManifest, TaskSubmission, BenchmarkRun
from .service import ServiceConfig, SuperAIService
from .storage import AccessDenied, ArtifactNotFound, IntegrityError
from .hive import CapacityError, HiveTransitionError
from .execution import PlanningError
from .learning import LearningSafetyError


class SourceImportRequest(BaseModel):
    title: str = Field(min_length=1, max_length=512)
    text: str = Field(min_length=1, max_length=2_000_000)
    project_id: Optional[str] = None
    visibility: str = Field(default="tenant", pattern="^(tenant|project|global)$")
    sectors: list[str] = Field(default_factory=list)
    trusted: bool = True

    @model_validator(mode="after")
    def project_scope_requires_project_id(self) -> "SourceImportRequest":
        if self.visibility == "project" and not self.project_id:
            raise ValueError("project_id is required for project visibility")
        return self


class SkillCompileRequest(BaseModel):
    train_task_ids: list[str] = Field(min_length=2)
    holdout_task_ids: list[str] = Field(default_factory=list)


class SkillValidationRequest(BaseModel):
    quality_delta: float
    latency_delta: float = 0.0
    resource_delta: float = 0.0
    risk_penalty: float = 0.0


class DatasetTrainingRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=1000)
    project_id: Optional[str] = None
    visibility: str = Field(default="tenant", pattern="^(tenant|project|global)$")
    sectors: list[str] = Field(default_factory=list)
    trusted: bool = True

    @model_validator(mode="after")
    def project_scope_requires_project_id(self) -> "DatasetTrainingRequest":
        if self.visibility == "project" and not self.project_id:
            raise ValueError("project_id is required for project visibility")
        return self


def create_app(config: Optional[ServiceConfig] = None) -> FastAPI:
    service = SuperAIService(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service
        try:
            yield
        finally:
            service.close()

    app = FastAPI(title="SuperAI API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(KeyError)
    async def not_found(_: object, exc: KeyError) -> Response:
        return Response(
            content=ErrorEnvelope(code="not_found", message=str(exc)).model_dump_json(),
            status_code=404,
            media_type="application/json",
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: object, exc: RequestValidationError) -> Response:
        return Response(
            content=ErrorEnvelope(
                code="validation_error",
                message="Request does not match the versioned API contract.",
                details={"errors": exc.errors()},
            ).model_dump_json(),
            status_code=422,
            media_type="application/json",
        )

    async def storage_error(_: object, exc: Exception) -> Response:
        status = 403 if isinstance(exc, AccessDenied) else 404 if isinstance(exc, ArtifactNotFound) else 409
        return Response(
            content=ErrorEnvelope(code="access_denied" if status == 403 else "artifact_error", message=str(exc)).model_dump_json(),
            status_code=status,
            media_type="application/json",
        )

    app.add_exception_handler(AccessDenied, storage_error)
    app.add_exception_handler(ArtifactNotFound, storage_error)
    app.add_exception_handler(IntegrityError, storage_error)

    async def domain_error(_: object, exc: Exception) -> Response:
        return Response(
            content=ErrorEnvelope(code="domain_error", message=str(exc)).model_dump_json(),
            status_code=409,
            media_type="application/json",
        )

    app.add_exception_handler(CapacityError, domain_error)
    app.add_exception_handler(HiveTransitionError, domain_error)
    app.add_exception_handler(PlanningError, domain_error)
    app.add_exception_handler(LearningSafetyError, domain_error)

    @app.get("/api/v1/health")
    def health() -> dict:
        return service.health()

    @app.get("/api/v1/meta")
    def meta() -> dict:
        return service.meta()

    @app.post("/api/v1/tasks", status_code=202)
    def submit_task(
        payload: TaskSubmission,
        idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
        tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-Id"),
    ) -> dict:
        # Local development may still pass tenant_id in the body. Once a
        # transport auth context is supplied, it is authoritative.
        if tenant_id is not None:
            payload = payload.model_copy(update={"tenant_id": tenant_id})
        # HTTP submission is non-blocking: the durable worker updates status
        # and the client polls the same task DTO with backoff.
        return service.submit_task(payload, idempotency_key=idempotency_key, execute_now=False).model_dump(mode="json")

    @app.post("/api/v1/tasks/{task_id}/cancel")
    def cancel_task(
        task_id: str,
        project_id: Optional[str] = Query(default=None),
        tenant_id: str = Header(default="local", alias="X-Tenant-Id"),
    ) -> dict:
        return service.cancel_task(task_id, tenant_id, project_id).model_dump(mode="json")

    @app.get("/api/v1/tasks/{task_id}")
    def task(
        task_id: str,
        project_id: Optional[str] = Query(default=None),
        tenant_id: str = Header(default="local", alias="X-Tenant-Id"),
    ) -> dict:
        return service.task(task_id, tenant_id, project_id, enforce_project=True).model_dump(mode="json")

    @app.get("/api/v1/traces/{trace_id}")
    def trace(
        trace_id: str,
        project_id: Optional[str] = Query(default=None),
        tenant_id: str = Header(default="local", alias="X-Tenant-Id"),
    ) -> dict:
        return service.trace(trace_id, tenant_id, project_id)

    @app.get("/api/v1/hives/{hive_id}")
    def hive(
        hive_id: str,
        project_id: Optional[str] = Query(default=None),
        tenant_id: str = Header(default="local", alias="X-Tenant-Id"),
    ) -> dict:
        return service.hive(hive_id, tenant_id, project_id).model_dump(mode="json")

    @app.post("/api/v1/hives/{hive_id}/freeze")
    def freeze_hive(
        hive_id: str,
        task_id: str,
        trace_id: str,
        project_id: Optional[str] = Query(default=None),
        tenant_id: str = Header(default="local", alias="X-Tenant-Id"),
    ) -> dict:
        task = service.task(task_id, tenant_id, project_id, enforce_project=True)
        if task.hive_id != hive_id or task.trace_id != trace_id:
            raise HiveTransitionError("freeze command does not match the task's hive and trace")
        service.hive(hive_id, tenant_id, project_id)
        return service.hives.freeze(hive_id, tenant_id, trace_id).model_dump(mode="json")

    @app.post("/api/v1/hives/{hive_id}/restore")
    def restore_hive(
        hive_id: str,
        task_id: str,
        trace_id: str,
        project_id: Optional[str] = Query(default=None),
        tenant_id: str = Header(default="local", alias="X-Tenant-Id"),
    ) -> dict:
        task = service.task(task_id, tenant_id, project_id, enforce_project=True)
        if task.hive_id != hive_id or task.trace_id != trace_id:
            raise HiveTransitionError("restore command does not match the task's hive and trace")
        service.hive(hive_id, tenant_id, project_id)
        return service.hives.restore(hive_id, tenant_id, trace_id).model_dump(mode="json")

    @app.get("/api/v1/artifacts/{artifact_id}/metadata")
    def artifact_metadata(
        artifact_id: str,
        project_id: Optional[str] = Query(default=None),
        tenant_id: str = Header(default="local", alias="X-Tenant-Id"),
    ) -> dict:
        return service.store.get_metadata(artifact_id, tenant_id, project_id=project_id).model_dump(mode="json")

    @app.post("/api/v1/sources")
    def import_source(
        payload: SourceImportRequest,
        tenant_id: str = Header(default="local", alias="X-Tenant-Id"),
        allow_global_publish: bool = Header(default=False, alias="X-Allow-Global-Publish"),
        allow_trusted_import: bool = Header(default=True, alias="X-Allow-Trusted-Import"),
    ) -> dict:
        if payload.visibility == "global" and not allow_global_publish:
            raise AccessDenied("global publication requires explicit policy approval")
        if payload.trusted and not allow_trusted_import:
            raise AccessDenied("trusted import requires explicit policy approval")
        scope = AccessScope(tenant_id=tenant_id, project_id=payload.project_id, visibility=payload.visibility)
        return service.cosmos.import_text(
            title=payload.title,
            text=payload.text,
            tenant_id=tenant_id,
            access_scope=scope,
            sectors=payload.sectors,
            trusted=payload.trusted,
        ).model_dump(mode="json")

    @app.delete("/api/v1/sources/{source_id}")
    def delete_source(source_id: str, tenant_id: str = Header(default="local", alias="X-Tenant-Id")) -> dict:
        return {"source_id": source_id, "removed_claims": service.cosmos.delete_source(source_id, tenant_id)}

    @app.get("/api/v1/cosmos/concepts")
    def concepts(
        query: Optional[str] = None,
        project_id: Optional[str] = None,
        limit: int = Query(default=100, ge=1, le=500),
        tenant_id: str = Header(default="local", alias="X-Tenant-Id"),
    ) -> list[dict]:
        return [item.model_dump(mode="json") for item in service.cosmos.list_concepts(tenant_id=tenant_id, project_id=project_id, query=query, limit=limit)]

    @app.get("/api/v1/cosmos/claims")
    def claims(
        project_id: Optional[str] = None,
        limit: int = Query(default=100, ge=1, le=500),
        tenant_id: str = Header(default="local", alias="X-Tenant-Id"),
    ) -> list[dict]:
        return [item.model_dump(mode="json") for item in service.cosmos.list_claims(tenant_id=tenant_id, project_id=project_id, limit=limit)]

    @app.get("/api/v1/atlas/capabilities")
    def capabilities() -> list[dict]:
        return [item.model_dump(mode="json") for item in service.atlas.manifests()]

    @app.get("/api/v1/system/dead-letters")
    def dead_letters(tenant_id: str = Header(default="local", alias="X-Tenant-Id")) -> list[dict]:
        return [item.model_dump(mode="json") for item in service.runtime.dead_letters(tenant_id)]

    @app.post("/api/v1/tasks/{task_id}/compost")
    def decompose_task(
        task_id: str,
        project_id: Optional[str] = Query(default=None),
        tenant_id: str = Header(default="local", alias="X-Tenant-Id"),
    ) -> dict:
        return service.learning.decompose_task(task_id, tenant_id, project_id=project_id).model_dump(mode="json")

    @app.post("/api/v1/compost/{compost_id}/validate")
    def validate_compost(
        compost_id: str,
        project_id: Optional[str] = Query(default=None),
        tenant_id: str = Header(default="local", alias="X-Tenant-Id"),
    ) -> dict:
        return service.learning.validate_compost(compost_id, tenant_id, project_id=project_id).model_dump(mode="json")

    @app.post("/api/v1/compost/{compost_id}/integrate")
    def integrate_compost(
        compost_id: str,
        project_id: Optional[str] = Query(default=None),
        tenant_id: str = Header(default="local", alias="X-Tenant-Id"),
    ) -> dict:
        return service.learning.integrate_compost(compost_id, tenant_id, project_id=project_id).model_dump(mode="json")

    @app.post("/api/v1/skills")
    def compile_skill(payload: SkillCompileRequest, tenant_id: str = Header(default="local", alias="X-Tenant-Id")) -> dict:
        return service.learning.compile_candidate(
            tenant_id=tenant_id,
            train_task_ids=payload.train_task_ids,
            holdout_task_ids=payload.holdout_task_ids,
        ).model_dump(mode="json")

    @app.get("/api/v1/skills")
    def skills(
        project_id: Optional[str] = Query(default=None),
        tenant_id: str = Header(default="local", alias="X-Tenant-Id"),
    ) -> list[dict]:
        return [skill.model_dump(mode="json") for skill in service.learning.skills(tenant_id=tenant_id, project_id=project_id)]

    @app.post("/api/v1/skills/{skill_id}/{version}/validate")
    def validate_skill(
        skill_id: str,
        version: str,
        payload: SkillValidationRequest,
        project_id: Optional[str] = Query(default=None),
        tenant_id: str = Header(default="local", alias="X-Tenant-Id"),
    ) -> dict:
        return service.learning.validate_skill(
            skill_id, version, tenant_id=tenant_id, project_id=project_id, **payload.model_dump()
        ).model_dump(mode="json")

    @app.post("/api/v1/skills/{skill_id}/{version}/shadow")
    def shadow_skill(
        skill_id: str,
        version: str,
        project_id: Optional[str] = Query(default=None),
        tenant_id: str = Header(default="local", alias="X-Tenant-Id"),
    ) -> dict:
        return service.learning.shadow_skill(skill_id, version, tenant_id=tenant_id, project_id=project_id).model_dump(mode="json")

    @app.post("/api/v1/skills/{skill_id}/{version}/activate")
    def activate_skill(
        skill_id: str,
        version: str,
        project_id: Optional[str] = Query(default=None),
        tenant_id: str = Header(default="local", alias="X-Tenant-Id"),
    ) -> dict:
        return service.learning.activate_skill(skill_id, version, tenant_id=tenant_id, project_id=project_id).model_dump(mode="json")

    @app.post("/api/v1/training/dataset")
    def train_dataset(
        payload: DatasetTrainingRequest,
        tenant_id: str = Header(default="local", alias="X-Tenant-Id"),
        allow_global_publish: bool = Header(default=False, alias="X-Allow-Global-Publish"),
        allow_trusted_import: bool = Header(default=True, alias="X-Allow-Trusted-Import"),
    ) -> dict:
        if payload.visibility == "global" and not allow_global_publish:
            raise AccessDenied("global publication requires explicit policy approval")
        if payload.trusted and not allow_trusted_import:
            raise AccessDenied("trusted import requires explicit policy approval")
        scope = AccessScope(tenant_id=tenant_id, project_id=payload.project_id, visibility=payload.visibility)
        results = []
        for text in payload.texts:
            result = service.cosmos.import_text(
                title=f"dataset-training-{len(results)+1}",
                text=text,
                tenant_id=tenant_id,
                access_scope=scope,
                sectors=payload.sectors,
                trusted=payload.trusted,
            )
            results.append(result.model_dump(mode="json"))
        claims = service.cosmos.list_claims(tenant_id=tenant_id, project_id=payload.project_id, limit=500)
        concepts = service.cosmos.list_concepts(tenant_id=tenant_id, project_id=payload.project_id, limit=500)
        return {
            "processed": len(results),
            "sources": results,
            "imported_claims": sum(item["imported_claims"] for item in results),
            "imported_concepts": sum(item["imported_concepts"] for item in results),
            "duplicates": sum(1 for item in results if item["duplicate"]),
            "visualization": {
                "concept_count": len(concepts),
                "claim_count": len(claims),
                "latest_source_ids": [item["source_id"] for item in results],
            },
        }

    @app.get("/api/v1/omega/tasks/{task_id}")
    def omega_task_state(
        task_id: str,
        project_id: Optional[str] = Query(default=None),
        tenant_id: str = Header(default="local", alias="X-Tenant-Id"),
    ) -> dict:
        service.task(task_id, tenant_id, project_id, enforce_project=True)
        snapshot = service.database.one(
            "SELECT * FROM active_graph_snapshots WHERE task_id = ? AND tenant_id = ? ORDER BY created_at DESC LIMIT 1",
            (task_id, tenant_id),
        )
        hypotheses = service.database.all(
            "SELECT * FROM hypotheses WHERE task_id = ? AND tenant_id = ? ORDER BY created_at",
            (task_id, tenant_id),
        )
        return {"snapshot": snapshot, "hypotheses": hypotheses}

    @app.post("/api/v1/genomes")
    def register_genome(payload: GenomeManifest) -> dict:
        return service.genomes.register(payload).model_dump(mode="json")

    @app.get("/api/v1/genomes/{genome_id}/{version}")
    def materialize_genome(genome_id: str, version: str) -> dict:
        return service.genomes.materialize(genome_id, version).model_dump(mode="json")

    # Benchmark endpoints
    @app.get("/api/v1/benchmarks/{run_id}")
    def benchmark_run(
        run_id: str,
        project_id: Optional[str] = Query(default=None),
        tenant_id: str = Header(default="local", alias="X-Tenant-Id"),
    ) -> dict:
        row = service.database.one(
            "SELECT * FROM benchmark_runs WHERE run_id = ? AND tenant_id = ?",
            (run_id, tenant_id),
        )
        if not row:
            raise KeyError("benchmark run not found")
        if row["project_id"] != project_id:
            raise AccessDenied("benchmark run belongs to another project")
        return BenchmarkRun(**row).model_dump(mode="json")

    @app.get("/api/v1/benchmarks")
    def list_benchmark_runs(
        project_id: Optional[str] = Query(default=None),
        mode: Optional[str] = Query(default=None),
        tenant_id: str = Header(default="local", alias="X-Tenant-Id"),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> list[dict]:
        query = "SELECT * FROM benchmark_runs WHERE tenant_id = ?"
        params = [tenant_id]
        if project_id:
            query += " AND project_id = ?"
            params.append(project_id)
        if mode:
            query += " AND mode = ?"
            params.append(mode)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = service.database.all(query, params)
        return [BenchmarkRun(**row).model_dump(mode="json") for row in rows]

    return app


app = create_app()
