"""FastAPI application for the canonical V2 model."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from server.database import init_db
from server.v2.hive import V2HiveService
from server.v2.physics import PlacementPhysicsV2
from server.v2.repository import V2Repository
from server.v2.training import TrainingPipelineV2
from server.v2.validation import ModelInvariantValidator


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="SuperAI Cloud / Space / Placement API",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    text: str = Field(..., min_length=1)


class HiveCreateRequest(BaseModel):
    max_cells: int = Field(default=24, ge=1, le=128)


class HiveQueryRequest(BaseModel):
    text: str = Field(..., min_length=1)


@app.get("/api/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "model": "cloud-space-placement", "version": "v2"}


@app.post("/api/v2/training/learn")
async def train(request: TrainRequest) -> Dict[str, Any]:
    return TrainingPipelineV2().train(request.text)


@app.get("/api/v2/field")
async def get_field() -> Dict[str, Any]:
    repository = V2Repository()
    with repository.transaction() as conn:
        space, _ = repository.get_or_create_space(conn, "global_field", seed=1337)
    return repository.normalized_space(int(space["id"]))


@app.get("/api/v2/stats")
async def get_stats() -> Dict[str, Any]:
    repository = V2Repository()
    with repository.transaction() as conn:
        return repository.stats(conn)


@app.delete("/api/v2/model")
async def clear_model() -> Dict[str, Any]:
    repository = V2Repository()
    repository.clear()
    with repository.transaction() as conn:
        return {"success": True, "stats": repository.stats(conn)}


@app.get("/api/v2/model")
async def get_model() -> Dict[str, Any]:
    return V2Repository().trained_model_snapshot()


@app.get("/api/v2/clouds/{cloud_id}")
async def get_cloud(cloud_id: int) -> Dict[str, Any]:
    cloud = V2Repository().get_cloud(cloud_id)
    if not cloud:
        raise HTTPException(status_code=404, detail="cloud not found")
    return {"cloud": cloud}


@app.get("/api/v2/clouds/{cloud_id}/structure")
async def get_structure(cloud_id: int) -> Dict[str, Any]:
    repository = V2Repository()
    with repository.transaction() as conn:
        cloud = conn.execute("SELECT * FROM clouds WHERE id = ?", (cloud_id,)).fetchone()
        if not cloud:
            raise HTTPException(status_code=404, detail="cloud not found")
        space = conn.execute(
            "SELECT * FROM spaces WHERE owner_cloud_id = ? AND space_type = 'word_structure_space'",
            (cloud_id,),
        ).fetchone()
        components = [dict(row) for row in conn.execute(
            """SELECT sc.* FROM structural_components sc
            WHERE sc.parent_cloud_id = ? ORDER BY sc.component_index""",
            (cloud_id,),
        )]
        child_ids = sorted({int(item["child_cloud_id"]) for item in components})
        children: Dict[str, Any] = {}
        if child_ids:
            marks = ",".join("?" for _ in child_ids)
            children = {
                str(row["id"]): dict(row)
                for row in conn.execute(f"SELECT * FROM clouds WHERE id IN ({marks})", child_ids)
            }
        return {
            "cloud": dict(cloud),
            "structure_space": dict(space) if space else None,
            "components": components,
            "clouds": children,
        }


@app.get("/api/v2/placements/{placement_id}")
async def get_placement(placement_id: int) -> Dict[str, Any]:
    repository = V2Repository()
    placement = repository.get_placement(placement_id)
    if not placement:
        raise HTTPException(status_code=404, detail="placement not found")
    cloud = repository.get_cloud(int(placement["cloud_id"]))
    return {"placement": placement, "cloud": cloud}


@app.get("/api/v2/spaces/{space_id}")
async def get_space(space_id: int) -> Dict[str, Any]:
    try:
        return V2Repository().normalized_space(space_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="space not found")


@app.post("/api/v2/spaces/{space_id}/physics/tick")
async def physics_tick(space_id: int) -> Dict[str, Any]:
    try:
        return {"space_id": space_id, "updates": PlacementPhysicsV2(space_id).tick()}
    except KeyError:
        raise HTTPException(status_code=404, detail="space not found")
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error))


@app.get("/api/v2/scenes/{scene_id}")
async def get_scene(scene_id: int) -> Dict[str, Any]:
    repository = V2Repository()
    with repository.transaction() as conn:
        scene = conn.execute("SELECT * FROM scenes WHERE cloud_id = ?", (scene_id,)).fetchone()
        if not scene:
            raise HTTPException(status_code=404, detail="scene not found")
        components = [
            {
                "id": row["id"],
                "placement_id": row["placement_id"],
                "cloud_id": row["word_form_cloud_id"],
                "lexeme_cloud_id": row["lexeme_cloud_id"],
                "token_index": row["token_index"],
                "grammatical_role": row["grammatical_role"],
                "dependency_role": row["dependency_role"],
                "head_component_id": row["head_component_id"],
                "confidence": row["confidence"],
                "morphology_json": row["morphology_json"],
            }
            for row in conn.execute(
                "SELECT * FROM scene_components WHERE scene_cloud_id = ? ORDER BY token_index",
                (scene_id,),
            )
        ]
        scene_dto = dict(scene)
        scene_dto["components"] = components
        return {"scene": scene_dto}


@app.post("/api/v2/hives")
async def create_hive(request: HiveCreateRequest) -> Dict[str, Any]:
    return V2HiveService().create(request.max_cells)


@app.get("/api/v2/hives/{hive_id}")
async def get_hive(hive_id: str) -> Dict[str, Any]:
    try:
        return V2HiveService().get_hive(hive_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="hive not found")


@app.post("/api/v2/hives/{hive_id}/query/preview")
async def preview_hive(hive_id: str, request: HiveQueryRequest) -> Dict[str, Any]:
    try:
        return V2HiveService().preview(hive_id, request.text)
    except KeyError:
        raise HTTPException(status_code=404, detail="hive not found")


@app.post("/api/v2/hives/{hive_id}/query")
async def query_hive(hive_id: str, request: HiveQueryRequest) -> Dict[str, Any]:
    try:
        return V2HiveService().query(hive_id, request.text)
    except KeyError:
        raise HTTPException(status_code=404, detail="hive not found")


@app.get("/api/v2/hives/{hive_id}/resonance-events")
async def hive_events(hive_id: str) -> Dict[str, Any]:
    try:
        return {"events": V2HiveService().service.events(hive_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="hive not found")


@app.get("/api/v2/hives/{hive_id}/search-decisions")
async def hive_decisions(hive_id: str) -> Dict[str, Any]:
    try:
        return {"decisions": V2HiveService().service.decisions(hive_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="hive not found")


@app.get("/api/v2/hives/{hive_id}/cells/{cell_id}/matches")
async def hive_matches(hive_id: str, cell_id: str) -> Dict[str, Any]:
    return {"matches": V2HiveService().service.matches(hive_id, cell_id)}


@app.get("/api/v2/debug/invariants")
async def invariants() -> Dict[str, Any]:
    return ModelInvariantValidator().validate()
