"""FastAPI server for the recursive nebula concept field."""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import uuid
import asyncio
import os

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from server.database import init_db, get_stats, reset_space, get_connection
from server.training import get_training_manager, TrainingManager
from server.repositories.cloud_repository import CloudRepository, SpaceRepository, LayerRepository, StructuralComponentRepository
from server.models.cloud import Cloud
from server.services.zoom import zoom_service
from server.services.lexeme import lexeme_service
from server.tokenizer import tokenize_hierarchical, TokenizationResult
from server.physics import PhysicsConfig
import json
from server.services.chat_session import chat_service
from server.v2.hive import V2HiveService
from server.v2.repository import V2Repository
from server.v2.training import TrainingPipelineV2
from server.v2.validation import ModelInvariantValidator


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="SuperAI Nebula API",
    description="Recursive multi-scale nebula system for semantic representation",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def model_schema_version() -> str:
    value = os.getenv("MODEL_SCHEMA_VERSION", "v2").casefold()
    return value if value in {"v1", "v2"} else "v1"


# ============================================================
# Request/Response Models
# ============================================================

class TrainRequest(BaseModel):
    text: str = Field(..., min_length=1)


class TrainResponse(BaseModel):
    success: bool
    concepts: List[Dict[str, Any]] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)
    time_ms: int = 0
    error: Optional[str] = None
    created_clouds: List[Dict[str, Any]] = Field(default_factory=list)
    strengthened_clouds: List[Dict[str, Any]] = Field(default_factory=list)
    new_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    position_changes: int = 0
    activations: List[Dict[str, Any]] = Field(default_factory=list)


class SpaceResponse(BaseModel):
    concepts: List[Dict[str, Any]] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)


class ResetResponse(BaseModel):
    success: bool
    concepts: List[Dict[str, Any]] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class LayerResponse(BaseModel):
    id: int
    name: str
    order_index: int
    scale: float
    layer_type: str
    config: Dict[str, Any]


class SpaceInfoResponse(BaseModel):
    space: Dict[str, Any]
    host_cloud: Optional[Dict[str, Any]] = None
    items: List[Dict[str, Any]] = Field(default_factory=list)
    breadcrumb: List[Dict[str, Any]] = Field(default_factory=list)


class CloudResponse(BaseModel):
    cloud: Dict[str, Any]
    spaces: List[Dict[str, Any]] = Field(default_factory=list)


class CloudChildrenResponse(BaseModel):
    parent_cloud: Dict[str, Any]
    children: List[Dict[str, Any]] = Field(default_factory=list)


class NeighborhoodResponse(BaseModel):
    clouds: List[Dict[str, Any]] = Field(default_factory=list)


class RegionSelectRequest(BaseModel):
    space_id: int
    x: float
    y: float
    radius: float


class RegionSelectResponse(BaseModel):
    candidates: List[Dict[str, Any]] = Field(default_factory=list)


class ActivateRequest(BaseModel):
    text: Optional[str] = None
    cloud_id: Optional[int] = None
    space_id: Optional[int] = None
    x: Optional[float] = None
    y: Optional[float] = None
    radius: Optional[float] = None


class ZoomInRequest(BaseModel):
    session_id: str
    cloud_id: int
    mode: str = "structural"  # structural or semantic


class ZoomOutRequest(BaseModel):
    session_id: str


class ChatMessageRequest(BaseModel):
    text: str = Field(..., min_length=1)


class ChatSessionResponse(BaseModel):
    session: Dict[str, Any]
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    hive: List[Dict[str, Any]] = Field(default_factory=list)
    turn: Optional[Dict[str, Any]] = None
    swarm: Dict[str, Any] = Field(default_factory=dict)


class HiveCreateRequest(BaseModel):
    max_cells: int = Field(default=24, ge=1, le=128)


class HiveQueryRequest(BaseModel):
    text: str = Field(..., min_length=1)


# ============================================================
# Health & Legacy Endpoints
# ============================================================

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "0.3.0", "model_schema_version": model_schema_version()}


# ============================================================
# Normalized Cloud / Space / Placement V2 API
# ============================================================

@app.post("/api/v2/training/learn")
async def train_v2(request: TrainRequest):
    return TrainingPipelineV2().train(request.text)


@app.get("/api/v2/field")
async def get_field_v2():
    repository = V2Repository()
    with repository.transaction() as conn:
        space = repository.ensure_space(conn, "global_field", seed=1337)
    return repository.normalized_space(int(space["id"]))


@app.get("/api/v2/clouds/{cloud_id}")
async def get_cloud_v2(cloud_id: int):
    cloud = V2Repository().get_cloud(cloud_id)
    if not cloud:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="cloud not found")
    return {"cloud": cloud}


@app.get("/api/v2/clouds/{cloud_id}/structure")
async def get_cloud_structure_v2(cloud_id: int):
    repository = V2Repository()
    with repository.transaction() as conn:
        cloud = conn.execute("SELECT * FROM v2_clouds WHERE id = ?", (cloud_id,)).fetchone()
        if not cloud:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="cloud not found")
        components = [dict(row) for row in conn.execute(
            """SELECT sc.*, child.canonical_name AS child_name, child.cloud_type AS child_type
            FROM v2_structural_components sc JOIN v2_clouds child ON child.id = sc.child_cloud_id
            WHERE sc.parent_cloud_id = ? ORDER BY sc.component_index""", (cloud_id,)
        ).fetchall()]
    return {"cloud": dict(cloud), "structural_components": components}


@app.get("/api/v2/spaces/{space_id}")
async def get_space_v2(space_id: int):
    try:
        return V2Repository().normalized_space(space_id)
    except KeyError:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="space not found")


@app.get("/api/v2/scenes/{scene_id}")
async def get_scene_v2(scene_id: int):
    repository = V2Repository()
    with repository.transaction() as conn:
        scene = conn.execute("SELECT * FROM v2_scenes WHERE cloud_id = ?", (scene_id,)).fetchone()
        if not scene:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="scene not found")
        components = [dict(row) for row in conn.execute(
            """SELECT sc.*, wf.canonical_name AS word_form, lx.canonical_name AS lexeme
            FROM v2_scene_components sc
            JOIN v2_clouds wf ON wf.id = sc.word_form_cloud_id
            JOIN v2_clouds lx ON lx.id = sc.lexeme_cloud_id
            WHERE sc.scene_cloud_id = ? ORDER BY sc.token_index""", (scene_id,)
        ).fetchall()]
    return {"scene": dict(scene), "scene_components": components}


@app.post("/api/v2/hives")
async def create_hive_v2(request: HiveCreateRequest):
    return V2HiveService().create(request.max_cells)


@app.post("/api/v2/hives/{hive_id}/query/preview")
async def preview_hive_query_v2(hive_id: str, request: HiveQueryRequest):
    try:
        return V2HiveService().preview(hive_id, request.text)
    except KeyError:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="hive not found")


@app.post("/api/v2/hives/{hive_id}/query")
async def query_hive_v2(hive_id: str, request: HiveQueryRequest):
    try:
        return V2HiveService().query(hive_id, request.text)
    except KeyError:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="hive not found")


@app.get("/api/v2/hives/{hive_id}")
async def get_hive_v2(hive_id: str):
    try:
        return V2HiveService().get_hive(hive_id)
    except KeyError:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="hive not found")


@app.get("/api/v2/hives/{hive_id}/resonance-events")
async def get_hive_resonance_events_v2(hive_id: str):
    try:
        return {"events": V2HiveService().service.events(hive_id)}
    except KeyError:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="hive not found")


@app.get("/api/v2/hives/{hive_id}/search-decisions")
async def get_hive_search_decisions_v2(hive_id: str):
    try:
        return {"decisions": V2HiveService().service.decisions(hive_id)}
    except KeyError:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="hive not found")


@app.get("/api/v2/hives/{hive_id}/cells/{cell_id}/matches")
async def get_hive_cell_matches_v2(hive_id: str, cell_id: str):
    try:
        return {"matches": V2HiveService().service.matches(hive_id, cell_id)}
    except KeyError:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="hive not found")


@app.get("/api/v2/debug/invariants")
async def get_v2_invariants():
    return ModelInvariantValidator().validate()


@app.post("/api/chat/sessions", response_model=ChatSessionResponse)
async def create_chat_session():
    return ChatSessionResponse(**chat_service.create_session())


@app.get("/api/chat/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(session_id: str):
    try:
        return ChatSessionResponse(**chat_service.get_state(session_id))
    except KeyError:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="chat session not found")


@app.post("/api/chat/sessions/{session_id}/messages")
async def send_chat_message(session_id: str, request: ChatMessageRequest):
    try:
        return chat_service.start_message(session_id, request.text)
    except (KeyError, ValueError):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="invalid chat session or message")


@app.delete("/api/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    chat_service.reset(session_id)
    return {"success": True}


@app.get("/api/chat/components")
async def inspect_chat_component(token: str = Query(..., min_length=1)):
    """Resolve a hive composition component to its real field hierarchy."""
    cloud_repo = CloudRepository()
    layer_repo = LayerRepository()
    component_repo = StructuralComponentRepository()
    matches: List[Dict[str, Any]] = []
    selected = None
    for layer_name in ("word_form", "lexeme", "concept", "scene"):
        layer = layer_repo.get_by_name(layer_name)
        if not layer:
            continue
        cloud = cloud_repo.get_by_canonical_name(layer.id, token.casefold())
        if not cloud:
            continue
        item = {"id": cloud.id, "label": cloud.canonical_name, "layer": layer_name, "type": cloud.cloud_type, "mass": cloud.mass, "activation": cloud.activation}
        matches.append(item)
        if selected is None or layer_name == "word_form":
            selected = cloud
    children: List[Dict[str, Any]] = []
    if selected:
        for component in component_repo.get_children(selected.id):
            child = cloud_repo.get_by_id(component.child_cloud_id)
            if child:
                child_layer = layer_repo.get_by_id(child.layer_id)
                children.append({"id": child.id, "label": child.canonical_name, "layer": child_layer.name if child_layer else "", "weight": component.weight, "position": component.position_index})
    return {"token": token, "matches": matches, "selected": next((item for item in matches if selected and item["id"] == selected.id), None), "children": children}


@app.post("/api/train", response_model=TrainResponse)
async def train_legacy(request: TrainRequest):
    return TrainResponse(**get_training_manager().learn(request.text))


@app.post("/api/v1/training/learn", response_model=TrainResponse)
async def train_v1(request: TrainRequest):
    return TrainResponse(**get_training_manager().learn(request.text))


@app.get("/api/space", response_model=SpaceResponse)
async def get_space_legacy():
    return SpaceResponse(**get_training_manager().get_space())


@app.get("/api/v1/training/space", response_model=SpaceResponse)
async def get_space_v1():
    return SpaceResponse(**get_training_manager().get_space())


@app.post("/api/reset", response_model=ResetResponse)
async def reset_legacy():
    return ResetResponse(**get_training_manager().reset_space())


@app.delete("/api/v1/training/space", response_model=ResetResponse)
async def reset_v1():
    return ResetResponse(**get_training_manager().reset_space())


# ============================================================
# Layer API
# ============================================================

@app.get("/api/layers", response_model=List[LayerResponse])
async def get_layers():
    """Get all available scale layers."""
    layer_repo = LayerRepository()
    layers = layer_repo.get_all_ordered()
    return [LayerResponse(**layer.to_dict()) for layer in layers]


@app.get("/api/layers/{layer_name}", response_model=LayerResponse)
async def get_layer(layer_name: str):
    """Get layer by name."""
    layer_repo = LayerRepository()
    layer = layer_repo.get_by_name(layer_name)
    if not layer:
        return {"error": "Layer not found"}
    return LayerResponse(**layer.to_dict())


# ============================================================
# Space API
# ============================================================

@app.get("/api/spaces/{space_id}", response_model=SpaceInfoResponse)
async def get_space(space_id: int):
    """Get space details with visible clouds."""
    space_repo = SpaceRepository()
    space = space_repo.get_by_id(space_id)
    if not space:
        return {"error": "Space not found"}
    
    cloud_repo = CloudRepository()
    placement_repo = cloud_repo  # placeholder
    
    # Get placements in this space
    from server.repositories.cloud_repository import CloudPlacementRepository
    placement_repo = CloudPlacementRepository()
    placements = placement_repo.get_by_space(space_id)
    
    items = []
    for p in placements:
        cloud = cloud_repo.get_by_id(p.cloud_id)
        if cloud:
            items.append({
                "cloud": cloud.to_dict(),
                "placement": p.to_dict(),
            })
    
    host_cloud = cloud_repo.get_by_id(space.host_cloud_id) if space.host_cloud_id else None
    
    return SpaceInfoResponse(
        space=space.to_dict(),
        host_cloud=host_cloud.to_dict() if host_cloud else None,
        items=items,
    )


@app.get("/api/spaces/{space_id}/clouds")
async def get_space_clouds(
    space_id: int,
    min_x: float = Query(-10000),
    min_y: float = Query(-10000),
    max_x: float = Query(10000),
    max_y: float = Query(10000),
    limit: int = Query(1000),
    active_only: bool = Query(False),
    min_density: float = Query(0.0),
):
    """Get clouds in a space, optionally filtered by viewport."""
    from server.repositories.cloud_repository import CloudPlacementRepository
    
    cloud_repo = CloudRepository()
    placement_repo = CloudPlacementRepository()
    
    if min_x > -10000 or max_x < 10000 or min_y > -10000 or max_y < 10000:
        # Viewport query
        placements = placement_repo.get_in_viewport(space_id, min_x, min_y, max_x, max_y, min_density)
    else:
        placements = placement_repo.get_by_space(space_id)
        if min_density > 0:
            placements = [p for p in placements if p.density >= min_density]
        if active_only:
            placements = [p for p in placements if p.activation > 0.1]
    
    # Limit results
    placements = placements[:limit]
    
    result = []
    for p in placements:
        cloud = cloud_repo.get_by_id(p.cloud_id)
        if cloud:
            result.append({
                "cloud": cloud.to_dict(),
                "placement": p.to_dict(),
            })
    
    return {"clouds": result, "count": len(result)}


# ============================================================
# Cloud API
# ============================================================

@app.get("/api/clouds/{cloud_id}", response_model=CloudResponse)
async def get_cloud(cloud_id: int):
    """Get global cloud info and its available spaces."""
    cloud_repo = CloudRepository()
    cloud = cloud_repo.get_by_id(cloud_id)
    if not cloud:
        return {"error": "Cloud not found"}
    
    space_repo = SpaceRepository()
    spaces = space_repo.get_by_host_cloud(cloud_id)
    
    return CloudResponse(
        cloud=cloud.to_dict(),
        spaces=[s.to_dict() for s in spaces],
    )


@app.get("/api/clouds/{cloud_id}/spaces")
async def get_cloud_spaces(cloud_id: int):
    """Get available spaces for a cloud (structural and semantic)."""
    space_repo = SpaceRepository()
    structural = space_repo.get_structural_space(cloud_id)
    semantic = space_repo.get_semantic_space(cloud_id)
    
    return {
        "structural": structural.to_dict() if structural else None,
        "semantic": semantic.to_dict() if semantic else None,
    }


@app.get("/api/clouds/{cloud_id}/children", response_model=CloudChildrenResponse)
async def get_cloud_children(cloud_id: int):
    """Get structural children of a cloud (for zoom-in)."""
    cloud_repo = CloudRepository()
    component_repo = CloudRepository()  # placeholder
    from server.repositories.cloud_repository import StructuralComponentRepository
    component_repo = StructuralComponentRepository()
    
    cloud = cloud_repo.get_by_id(cloud_id)
    if not cloud:
        return {"error": "Cloud not found"}
    
    components = component_repo.get_children(cloud_id)
    
    children = []
    for comp in components:
        child_cloud = cloud_repo.get_by_id(comp.child_cloud_id)
        if child_cloud:
            children.append({
                "component": comp.to_dict(),
                "cloud": child_cloud.to_dict(),
            })
    
    return CloudChildrenResponse(
        parent_cloud=cloud.to_dict(),
        children=children,
    )


@app.get("/api/clouds/{cloud_id}/neighborhood", response_model=NeighborhoodResponse)
async def get_cloud_neighborhood(cloud_id: int, layer_name: str = "concept", limit: int = 20):
    """Get geometrically close clouds in the same layer (no semantic edges)."""
    from server.services.activation import get_coactivation_neighbors
    from server.repositories.cloud_repository import LayerRepository
    
    cloud_repo = CloudRepository()
    layer_repo = LayerRepository()
    
    cloud = cloud_repo.get_by_id(cloud_id)
    if not cloud:
        return {"error": "Cloud not found"}
    
    layer = layer_repo.get_by_name(layer_name)
    if not layer:
        return {"error": "Layer not found"}
    
    # Get co-activation neighbors
    neighbors = get_coactivation_neighbors(cloud_id, layer.id, min_score=0.01, limit=limit)
    
    result = []
    for nid, score in neighbors:
        n_cloud = cloud_repo.get_by_id(nid)
        if n_cloud:
            result.append({
                "cloud": n_cloud.to_dict(),
                "coactivation_score": score,
            })
    
    return NeighborhoodResponse(clouds=result)


# ============================================================
# Zoom/Navigation API
# ============================================================

@app.post("/api/zoom/in/structural", response_model=SpaceInfoResponse)
async def zoom_in_structural(cloud_id: int, session_id: str = "default"):
    """Zoom into cloud's structural space (lower layer composition)."""
    result = zoom_service.zoom_in_structural(session_id, cloud_id)
    if not result:
        return {"error": "Cloud not found"}
    return SpaceInfoResponse(**result)


@app.post("/api/zoom/in/semantic", response_model=SpaceInfoResponse)
async def zoom_in_semantic(cloud_id: int, session_id: str = "default"):
    """Zoom into cloud's semantic space (same layer neighbors)."""
    result = zoom_service.zoom_in_semantic(session_id, cloud_id)
    if not result:
        return {"error": "Cloud not found"}
    return SpaceInfoResponse(**result)


@app.post("/api/zoom/out", response_model=SpaceInfoResponse)
async def zoom_out(session_id: str = "default"):
    """Zoom out to parent space."""
    result = zoom_service.zoom_out(session_id)
    if not result:
        return {"error": "No active zoom session"}
    return SpaceInfoResponse(**result)


@app.get("/api/zoom/current", response_model=SpaceInfoResponse)
async def get_current_zoom(session_id: str = "default"):
    """Get current zoom state."""
    result = zoom_service.get_current_space(session_id)
    if not result:
        return {"at_root": True, "breadcrumb": []}
    return SpaceInfoResponse(**result)


@app.get("/api/zoom/breadcrumb")
async def get_breadcrumb(session_id: str = "default"):
    """Get navigation breadcrumb."""
    path = zoom_service.get_or_create_path(session_id)
    return {"breadcrumb": path.spaces, "current_space_id": path.current_space_id, "mode": path.mode}


# ============================================================
# Region Selection API
# ============================================================

@app.post("/api/select-region", response_model=RegionSelectResponse)
async def select_region(request: RegionSelectRequest):
    """
    Select a region in a space and return top-K candidate clouds
    by density at that point. Multiple clouds can overlap.
    """
    from server.repositories.cloud_repository import CloudPlacementRepository, CloudRepository
    from server.physics import create_space_physics
    
    placement_repo = CloudPlacementRepository()
    cloud_repo = CloudRepository()
    
    # Get placements in region
    placements = placement_repo.get_in_viewport(
        request.space_id,
        request.x - request.radius,
        request.y - request.radius,
        request.x + request.radius,
        request.y + request.radius,
        min_density=0.0
    )
    
    # Compute density at point for each cloud
    from server.services.spatial_index import compute_density_at_point
    
    candidates = []
    for p in placements:
        cloud = cloud_repo.get_by_id(p.cloud_id)
        if not cloud:
            continue
        
        # Local density at query point
        local_density = p.mass * 1.0  # simplified
        
        # Could compute actual Gaussian density here
        # For now use placement density * activation
        score = p.density * max(p.activation, 0.1) * cloud.stability
        
        candidates.append({
            "cloud": cloud.to_dict(),
            "placement": p.to_dict(),
            "score": score,
            "local_density": p.density,
        })
    
    # Sort by score
    candidates.sort(key=lambda c: c["score"], reverse=True)
    
    return RegionSelectResponse(candidates=candidates[:10])


# ============================================================
# Tokenization API
# ============================================================

@app.post("/api/tokenize")
async def tokenize_text(request: TrainRequest):
    """Hierarchical tokenization: text -> sentences -> words -> characters."""
    result = tokenize_hierarchical(request.text)
    return result.to_dict()


# ============================================================
# Activation API
# ============================================================

@app.post("/api/activate")
async def activate(request: ActivateRequest):
    """Activate clouds by text, cloud ID, or spatial region."""
    from server.services.activation import ActivationManager, compute_activation_from_text
    
    manager = ActivationManager()
    
    if request.text:
        # Activate from text
        word_activations = compute_activation_from_text(request.text, "word_form")
        char_activations = compute_activation_from_text(request.text, "character")
        
        cloud_repo = CloudRepository()
        activated = []
        
        for cloud_id, value in word_activations.items():
            cloud = cloud_repo.get_by_id(cloud_id)
            if cloud:
                manager.activate_cloud(cloud, value)
                cloud_repo.increment_observation(cloud_id, mass_delta=0.05, stability_delta=0.01)
                activated.append({"cloud_id": cloud_id, "value": value, "layer": "word_form"})
        
        for cloud_id, value in char_activations.items():
            cloud = cloud_repo.get_by_id(cloud_id)
            if cloud:
                manager.activate_cloud(cloud, value)
                cloud_repo.increment_observation(cloud_id, mass_delta=0.02, stability_delta=0.005)
                activated.append({"cloud_id": cloud_id, "value": value, "layer": "character"})
        
        return {"activated": activated, "count": len(activated)}
    
    if request.cloud_id:
        cloud_repo = CloudRepository()
        cloud = cloud_repo.get_by_id(request.cloud_id)
        if cloud:
            manager.activate_cloud(cloud, 1.0)
            cloud_repo.increment_observation(cloud_id, mass_delta=0.1, stability_delta=0.02)
            return {"activated": [{"cloud_id": cloud_id, "value": 1.0}]}
    
    return {"activated": [], "count": 0}


# ============================================================
# Field Hierarchy API
# ============================================================

class HierarchyResponse(BaseModel):
    scenes: List[Dict[str, Any]] = Field(default_factory=list)
    structural_spaces: List[Dict[str, Any]] = Field(default_factory=list)
    lexemes: List[Dict[str, Any]] = Field(default_factory=list)
    semantic_overlays: List[Dict[str, Any]] = Field(default_factory=list)


@app.get("/api/field/hierarchy", response_model=HierarchyResponse)
async def get_field_hierarchy(max_depth: int = Query(default=3, ge=1, le=4)):
    result = HierarchyResponse()
    word_occurrences: List[Dict[str, Any]] = []

    with get_connection() as conn:
        layer_rows = conn.execute("SELECT id, name FROM layers").fetchall()
        layers = {row["name"]: int(row["id"]) for row in layer_rows}
        scene_layer_id = layers.get("scene")
        if not scene_layer_id:
            return result

        scene_rows = conn.execute(
            """SELECT s.*, c.canonical_name, c.mass, c.density, c.radius, c.stability,
                c.activation, c.observation_count,
                COALESCE(p.x, 500) AS x, COALESCE(p.y, 350) AS y
            FROM scenes s
            JOIN clouds c ON c.id = s.scene_cloud_id
            LEFT JOIN spaces gs ON gs.layer_id = c.layer_id AND gs.mode = 'global' AND gs.host_cloud_id = 0
            LEFT JOIN cloud_placements p ON p.space_id = gs.id AND p.cloud_id = c.id
            ORDER BY c.mass DESC, s.id"""
        ).fetchall()

        for scene_row in scene_rows:
            word_ids = [int(item) for item in json.loads(scene_row["word_form_cloud_ids_json"] or "[]")]
            lexeme_ids = [int(item) for item in json.loads(scene_row["lexeme_ids_json"] or "[]")]
            placement_rows = conn.execute(
                """SELECT sc.position_index, sc.child_cloud_id, p.x, p.y, p.radius
                FROM structural_components sc
                LEFT JOIN cloud_placements p ON p.id = sc.child_placement_id
                WHERE sc.parent_cloud_id = ? AND sc.role = 'word_form'
                ORDER BY sc.position_index""",
                (scene_row["scene_cloud_id"],),
            ).fetchall()
            placements = {int(row["position_index"]): row for row in placement_rows}
            scene_words: List[Dict[str, Any]] = []
            for index, word_id in enumerate(word_ids):
                word_row = conn.execute(
                    """SELECT c.*, l.id AS lexeme_id, l.canonical_form, l.pos_tag
                    FROM clouds c
                    LEFT JOIN word_form_to_lexeme w ON w.word_form_cloud_id = c.id
                    LEFT JOIN lexemes l ON l.id = w.lexeme_id
                    WHERE c.id = ? ORDER BY w.is_canonical DESC LIMIT 1""",
                    (word_id,),
                ).fetchone()
                if not word_row:
                    continue
                placement = placements.get(index)
                local_x = float(placement["x"]) if placement and placement["x"] is not None else 0.0
                local_y = float(placement["y"]) if placement and placement["y"] is not None else 0.0
                world_x = float(scene_row["x"]) + local_x
                world_y = float(scene_row["y"]) + local_y
                lexeme_id = (
                    int(word_row["lexeme_id"])
                    if word_row["lexeme_id"] is not None
                    else (lexeme_ids[index] if index < len(lexeme_ids) else None)
                )
                characters: List[Dict[str, Any]] = []
                if max_depth >= 3:
                    character_rows = conn.execute(
                        """SELECT sc.position_index, c.id, c.canonical_name, c.mass,
                            c.density, c.radius, c.stability, c.activation, p.x, p.y
                        FROM structural_components sc
                        JOIN clouds c ON c.id = sc.child_cloud_id
                        LEFT JOIN cloud_placements p ON p.id = sc.child_placement_id
                        WHERE sc.parent_cloud_id = ? AND sc.role = 'character'
                        ORDER BY sc.position_index""",
                        (word_id,),
                    ).fetchall()
                    for character in character_rows:
                        characters.append({
                            "id": int(character["id"]),
                            "key": f"{scene_row['id']}:{index}:{character['position_index']}",
                            "token": character["canonical_name"],
                            "index": int(character["position_index"]),
                            "x": world_x + float(character["x"] or 0.0),
                            "y": world_y + float(character["y"] or 0.0),
                            "radius": max(3.0, float(character["radius"]) * 0.55),
                            "mass": float(character["mass"]),
                            "density": float(character["density"]),
                            "stability": float(character["stability"]),
                            "activation": float(character["activation"]),
                            "layer": "character",
                            "cloud_type": "character",
                        })
                word = {
                    "id": int(word_row["id"]),
                    "key": f"{scene_row['id']}:{index}",
                    "token": word_row["canonical_name"],
                    "index": index,
                    "x": world_x,
                    "y": world_y,
                    "local_x": local_x,
                    "local_y": local_y,
                    "radius": max(28.0, float(placement["radius"]) * 2.4) if placement else 30.0,
                    "mass": float(word_row["mass"]),
                    "density": float(word_row["density"]),
                    "stability": float(word_row["stability"]),
                    "activation": float(word_row["activation"]),
                    "layer": "word_form",
                    "cloud_type": "word_form",
                    "lexeme_id": lexeme_id,
                    "lexeme": word_row["canonical_form"] or word_row["canonical_name"],
                    "pos_tag": word_row["pos_tag"],
                    "scene_cloud_id": int(scene_row["scene_cloud_id"]),
                    "scene_x": float(scene_row["x"]),
                    "scene_y": float(scene_row["y"]),
                    "scene_radius": float(scene_row["radius"]),
                    "characters": characters,
                }
                scene_words.append(word)
                word_occurrences.append(word)

            result.scenes.append({
                "id": int(scene_row["scene_cloud_id"]),
                "scene_id": int(scene_row["id"]),
                "token": scene_row["sentence_text"],
                "canonical_name": scene_row["canonical_name"],
                "sentence_text": scene_row["sentence_text"],
                "x": float(scene_row["x"]),
                "y": float(scene_row["y"]),
                "radius": float(scene_row["radius"]),
                "mass": float(scene_row["mass"]),
                "density": float(scene_row["density"]),
                "stability": float(scene_row["stability"]),
                "activation": float(scene_row["activation"]),
                "observation_count": int(scene_row["observation_count"]),
                "layer": "scene",
                "cloud_type": "scene",
                "words": scene_words,
                "word_forms": scene_words,
            })

        grouped_occurrences: Dict[int, List[Dict[str, Any]]] = {}
        for occurrence in word_occurrences:
            if occurrence["lexeme_id"] is not None:
                grouped_occurrences.setdefault(int(occurrence["lexeme_id"]), []).append(occurrence)
        for occurrences in grouped_occurrences.values():
            remaining = set(range(len(occurrences)))
            while remaining:
                component = {remaining.pop()}
                changed = True
                while changed:
                    changed = False
                    for candidate in list(remaining):
                        if any(
                            (
                                (occurrences[candidate]["scene_x"] - occurrences[index]["scene_x"]) ** 2
                                + (occurrences[candidate]["scene_y"] - occurrences[index]["scene_y"]) ** 2
                            ) ** 0.5
                            < occurrences[candidate]["scene_radius"] + occurrences[index]["scene_radius"]
                            for index in component
                        ):
                            component.add(candidate)
                            remaining.remove(candidate)
                            changed = True
                if len(component) < 2:
                    continue
                center_x = sum(occurrences[index]["x"] for index in component) / len(component)
                center_y = sum(occurrences[index]["y"] for index in component) / len(component)
                for index in component:
                    occurrence = occurrences[index]
                    dx = center_x - occurrence["x"]
                    dy = center_y - occurrence["y"]
                    occurrence["x"] = center_x
                    occurrence["y"] = center_y
                    for character in occurrence["characters"]:
                        character["x"] += dx
                        character["y"] += dy

        lexeme_rows = conn.execute(
            """SELECT l.*, c.id AS cloud_id, c.mass, c.density, c.radius,
                c.stability, c.activation
            FROM lexemes l
            LEFT JOIN clouds c ON c.canonical_name = l.canonical_form
                AND c.layer_id = ?
            ORDER BY l.frequency DESC, l.id""",
            (layers.get("lexeme", -1),),
        ).fetchall()
        for lexeme in lexeme_rows:
            forms = conn.execute(
                """SELECT c.id, c.canonical_name AS name FROM word_form_to_lexeme w
                JOIN clouds c ON c.id = w.word_form_cloud_id WHERE w.lexeme_id = ?""",
                (lexeme["id"],),
            ).fetchall()
            result.lexemes.append({
                "id": int(lexeme["cloud_id"] or lexeme["id"]),
                "lexeme_id": int(lexeme["id"]),
                "token": lexeme["canonical_form"],
                "canonical_form": lexeme["canonical_form"],
                "pos_tag": lexeme["pos_tag"],
                "frequency": int(lexeme["frequency"]),
                "mass": float(lexeme["mass"] or 1),
                "density": float(lexeme["density"] or 1),
                "radius": float(lexeme["radius"] or 18),
                "stability": float(lexeme["stability"] or 0),
                "activation": float(lexeme["activation"] or 0),
                "layer": "lexeme",
                "cloud_type": "lexeme",
                "word_forms": [dict(form) for form in forms],
            })

        concept_rows = conn.execute(
            """SELECT c.*, cc.member_lexeme_ids_json
            FROM clouds c
            JOIN concept_centroids cc ON cc.concept_cloud_id = c.id
            WHERE c.layer_id = ? ORDER BY c.mass DESC""",
            (layers.get("concept", -1),),
        ).fetchall()
        for concept in concept_rows:
            memberships = conn.execute(
                """SELECT m.lexeme_id, m.membership, m.centrality, l.canonical_form
                FROM lexeme_concept_membership m
                JOIN lexemes l ON l.id = m.lexeme_id
                WHERE m.concept_cloud_id = ? AND m.membership > 0
                ORDER BY m.membership DESC""",
                (concept["id"],),
            ).fetchall()
            weights = {int(row["lexeme_id"]): float(row["membership"]) for row in memberships}
            matched = [word for word in word_occurrences if word["lexeme_id"] in weights]
            if not matched:
                continue
            total_weight = sum(weights[word["lexeme_id"]] for word in matched)
            center_x = sum(word["x"] * weights[word["lexeme_id"]] for word in matched) / total_weight
            center_y = sum(word["y"] * weights[word["lexeme_id"]] for word in matched) / total_weight
            spread = max(
                (((word["x"] - center_x) ** 2 + (word["y"] - center_y) ** 2) ** 0.5)
                + word["radius"]
                for word in matched
            )
            result.semantic_overlays.append({
                "id": int(concept["id"]),
                "token": concept["canonical_name"],
                "concept_name": concept["canonical_name"],
                "center_x": center_x,
                "center_y": center_y,
                "radius": max(float(concept["radius"]), spread + 22.0),
                "mass": float(concept["mass"]),
                "density": float(concept["density"]),
                "stability": float(concept["stability"]),
                "activation": float(concept["activation"]),
                "layer": "concept",
                "cloud_type": "concept",
                "members": [
                    {
                        "lexeme_id": int(member["lexeme_id"]),
                        "canonical_form": member["canonical_form"],
                        "weight": float(member["membership"]),
                        "centrality": float(member["centrality"]),
                    }
                    for member in memberships
                ],
            })

        if max_depth >= 3:
            for scene in result.scenes:
                for word in scene["words"]:
                    result.structural_spaces.append({
                        "host_cloud": {"id": word["id"], "canonical_name": word["token"]},
                        "children": word["characters"],
                    })

    return result


# ============================================================
# Scene Similarity API
# ============================================================

@app.post("/api/scenes/similarity")
async def compute_scene_similarity(scene_a_id: int, scene_b_id: int):
    """Compute weighted Jaccard similarity between two scenes."""
    cloud_repo = CloudRepository()
    with get_connection() as conn:
        # Get scene data
        scene_a = conn.execute("SELECT * FROM scenes WHERE id = ?", (scene_a_id,)).fetchone()
        scene_b = conn.execute("SELECT * FROM scenes WHERE id = ?", (scene_b_id,)).fetchone()
        
        if not scene_a or not scene_b:
            return {"error": "Scene not found"}
        
        lexemes_a = set(json.loads(scene_a["lexeme_ids_json"] or "[]"))
        lexemes_b = set(json.loads(scene_b["lexeme_ids_json"] or "[]"))
        
        if not lexemes_a or not lexemes_b:
            return {"similarity": 0.0, "weight": 0.0}
        
        # Weighted Jaccard: (r1 + r2) * clamp(1.15 - 1.1 * similarity, 0.35, 1.15)
        intersection = lexemes_a & lexemes_b
        union = lexemes_a | lexemes_b
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Weight based on scene masses
        cloud_a = cloud_repo.get_by_id(scene_a["scene_cloud_id"])
        cloud_b = cloud_repo.get_by_id(scene_b["scene_cloud_id"])
        
        r1 = cloud_a.radius if cloud_a else 1.0
        r2 = cloud_b.radius if cloud_b else 1.0
        
        weight = (r1 + r2) * max(0.35, min(1.15, 1.15 - 1.1 * jaccard))
        
        # Store similarity
        conn.execute(
            """INSERT OR REPLACE INTO scene_similarity
            (scene_a_id, scene_b_id, similarity, weight, updated_at)
            VALUES (?, ?, ?, ?, ?)""",
            (scene_a_id, scene_b_id, jaccard, weight, now())
        )
        conn.commit()
        
        return {"similarity": jaccard, "weight": weight}


# ============================================================
# Statistics
# ============================================================

@app.get("/api/stats")
async def get_statistics():
    """Get system statistics."""
    stats = get_stats()
    
    # Add layer-specific stats
    cloud_repo = CloudRepository()
    layer_repo = LayerRepository()
    layers = layer_repo.get_all_ordered()
    
    layer_stats = []
    for layer in layers:
        clouds = cloud_repo.get_by_layer(layer.id, limit=10000)
        layer_stats.append({
            "layer": layer.to_dict(),
            "cloud_count": len(clouds),
            "total_mass": sum(c.mass for c in clouds),
            "avg_stability": sum(c.stability for c in clouds) / len(clouds) if clouds else 0,
        })
    
    return {
        "global": stats,
        "layers": layer_stats,
    }


# ============================================================
# WebSocket for Real-time Simulation
# ============================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
    
    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)
    
    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)
    
    async def broadcast(self, message: dict):
        for ws in self.active_connections.values():
            try:
                await ws.send_json(message)
            except:
                pass


manager = ConnectionManager()


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """Stream the actual bee algorithm and hive deposits for one chat session."""
    try:
        state = chat_service.get_state(session_id)
    except KeyError:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    async def listener(_session_id: str, event: Dict[str, Any]) -> None:
        await websocket.send_json(event)

    chat_service.subscribe(session_id, listener)
    try:
        await websocket.send_json({"type": "session_state", "sequence": 0, "payload": state})
        while True:
            message = await websocket.receive_json()
            if message.get("type") == "state":
                await websocket.send_json({"type": "session_state", "sequence": 0, "payload": chat_service.get_state(session_id)})
    except WebSocketDisconnect:
        pass
    finally:
        chat_service.unsubscribe(session_id, listener)


@app.websocket("/ws/simulation")
async def websocket_simulation(websocket: WebSocket):
    client_id = str(uuid.uuid4())[:8]
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Handle commands
            if data.get("type") == "subscribe_space":
                space_id = data.get("space_id")
                if space_id:
                    # Subscribe to space updates
                    await websocket.send_json({
                        "type": "subscribed",
                        "space_id": space_id,
                    })
            
            elif data.get("type") == "physics_tick":
                space_id = data.get("space_id")
                ticks = data.get("ticks", 1)
                
                if space_id:
                    # Run physics and send updates
                    from server.repositories.cloud_repository import CloudPlacementRepository, CloudRepository
                    placement_repo = CloudPlacementRepository()
                    cloud_repo = CloudRepository()
                    
                    placements = placement_repo.get_by_space(space_id)
                    clouds = {}
                    for p in placements:
                        cloud = cloud_repo.get_by_id(p.cloud_id)
                        if cloud:
                            clouds[cloud.id] = cloud
                    
                    from server.physics import create_space_physics
                    physics = create_space_physics(space_id, placements, clouds)
                    updates = physics.run_ticks(ticks)
                    
                    # Batch update
                    if updates:
                        placement_repo.update_positions_batch([
                            type('obj', (object,), {'id': pid, 'x': x, 'y': y})()
                            for pid, x, y in updates
                        ])
                    
                    await websocket.send_json({
                        "type": "physics_update",
                        "space_id": space_id,
                        "updates": [
                            {"placement_id": pid, "x": x, "y": y}
                            for pid, x, y in updates
                        ],
                    })
            
            elif data.get("type") == "get_density":
                space_id = data.get("space_id")
                x = data.get("x", 0)
                y = data.get("y", 0)
                
                if space_id:
                    from server.repositories.cloud_repository import CloudPlacementRepository, CloudRepository
                    placement_repo = CloudPlacementRepository()
                    cloud_repo = CloudRepository()
                    
                    placements = placement_repo.get_by_space(space_id)
                    clouds = {}
                    for p in placements:
                        cloud = cloud_repo.get_by_id(p.cloud_id)
                        if cloud:
                            clouds[cloud.id] = cloud
                    
                    from server.physics import create_space_physics
                    physics = create_space_physics(space_id, placements, clouds)
                    density = physics.query_density_at(x, y)
                    
                    await websocket.send_json({
                        "type": "density",
                        "x": x,
                        "y": y,
                        "density": density,
                    })
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
