"""FastAPI server for the recursive nebula concept field."""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from server.database import init_db, get_stats, reset_space, get_connection
from server.training import get_training_manager, TrainingManager
from server.repositories.cloud_repository import CloudRepository, SpaceRepository, LayerRepository
from server.models.cloud import Cloud
from server.services.zoom import zoom_service
from server.services.lexeme import lexeme_service
from server.tokenizer import tokenize_hierarchical, TokenizationResult
from server.physics import PhysicsConfig
import json


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


# ============================================================
# Health & Legacy Endpoints
# ============================================================

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "0.3.0"}


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
async def get_field_hierarchy(max_depth: int = 3):
    """
    Get field hierarchy with scenes, structural spaces, lexemes, and semantic overlays.
    Returns local centers, radii, and computed contributions for semantic overlays.
    """
    cloud_repo = CloudRepository()
    layer_repo = LayerRepository()
    space_repo = SpaceRepository()
    placement_repo = CloudPlacementRepository()
    
    # Get layer IDs
    scene_layer = layer_repo.get_by_name("scene")
    lexeme_layer = layer_repo.get_by_name("lexeme")
    concept_layer = layer_repo.get_by_name("concept")
    
    result = HierarchyResponse()
    
    # Get scenes
    if scene_layer:
        scene_clouds = cloud_repo.get_by_layer(scene_layer.id, limit=100)
        for cloud in scene_clouds:
            # Get scene details
            with get_connection() as conn:
                scene_row = conn.execute(
                    "SELECT * FROM scenes WHERE scene_cloud_id = ?", (cloud.id,)
                ).fetchone()
                
                if scene_row:
                    # Get word forms and lexemes for this scene
                    word_form_ids = json.loads(scene_row["word_form_cloud_ids_json"] or "[]")
                    lexeme_ids = json.loads(scene_row["lexeme_ids_json"] or "[]")
                    
                    word_forms = []
                    for wf_id in word_form_ids:
                        wf_cloud = cloud_repo.get_by_id(wf_id)
                        if wf_cloud:
                            word_forms.append(wf_cloud.to_dict())
                    
                    lexemes = []
                    for lex_id in lexeme_ids:
                        lex = lexeme_service.get_lexeme_for_word_form(lex_id)  # This won't work, need to fix
                        # Actually get lexeme by ID
                        with get_connection() as conn2:
                            lex_row = conn2.execute("SELECT * FROM lexemes WHERE id = ?", (lex_id,)).fetchone()
                            if lex_row:
                                lexemes.append({
                                    "id": lex_row["id"],
                                    "canonical_form": lex_row["canonical_form"],
                                    "pos_tag": lex_row["pos_tag"],
                                })
                    
                    result.scenes.append({
                        "cloud": cloud.to_dict(),
                        "sentence_text": scene_row["sentence_text"],
                        "word_forms": word_forms,
                        "lexemes": lexemes,
                    })
    
    # Get structural spaces (for word forms)
    word_form_layer = layer_repo.get_by_name("word_form")
    if word_form_layer:
        word_clouds = cloud_repo.get_by_layer(word_form_layer.id, limit=200)
        for cloud in word_clouds:
            struct_space = space_repo.get_structural_space(cloud.id)
            if struct_space:
                placements = placement_repo.get_by_space(struct_space.id)
                children = []
                for p in placements:
                    child = cloud_repo.get_by_id(p.cloud_id)
                    if child:
                        children.append({
                            "cloud": child.to_dict(),
                            "placement": p.to_dict(),
                        })
                result.structural_spaces.append({
                    "host_cloud": cloud.to_dict(),
                    "space": struct_space.to_dict(),
                    "children": children,
                })
    
    # Get lexemes
    if lexeme_layer:
        lexeme_clouds = cloud_repo.get_by_layer(lexeme_layer.id, limit=200)
        for cloud in lexeme_clouds:
            # Get lexeme details
            with get_connection() as conn:
                lex_row = conn.execute(
                    "SELECT * FROM lexemes WHERE canonical_form = ?", (cloud.canonical_form,)
                ).fetchone()
                if lex_row:
                    # Get word forms for this lexeme
                    wf_rows = conn.execute(
                        """SELECT wfl.word_form_cloud_id, c.canonical_name 
                        FROM word_form_to_lexeme wfl
                        JOIN clouds c ON wfl.word_form_cloud_id = c.id
                        WHERE wfl.lexeme_id = ?""",
                        (lex_row["id"],)
                    ).fetchall()
                    
                    word_forms = [{"id": r["word_form_cloud_id"], "name": r["canonical_name"]} for r in wf_rows]
                    
                    result.lexemes.append({
                        "cloud": cloud.to_dict(),
                        "lexeme": {
                            "id": lex_row["id"],
                            "canonical_form": lex_row["canonical_form"],
                            "pos_tag": lex_row["pos_tag"],
                            "frequency": lex_row["frequency"],
                        },
                        "word_forms": word_forms,
                    })
    
    # Get semantic overlays (concept projections)
    if concept_layer:
        concept_clouds = cloud_repo.get_by_layer(concept_layer.id, limit=100)
        for cloud in concept_clouds:
            # Get semantic space for this concept
            semantic_space = space_repo.get_semantic_space(cloud.id)
            if semantic_space:
                # Get overlays in this space
                with get_connection() as conn:
                    overlay_rows = conn.execute(
                        """SELECT * FROM semantic_overlays WHERE concept_cloud_id = ?""",
                        (cloud.id,)
                    ).fetchall()
                    
                    for overlay in overlay_rows:
                        member_lexeme_ids = json.loads(overlay["member_lexeme_ids_json"] or "[]")
                        member_weights = json.loads(overlay["member_weights_json"] or "[]")
                        
                        members = []
                        for lid, weight in zip(member_lexeme_ids, member_weights):
                            with get_connection() as conn2:
                                lex_row = conn2.execute("SELECT canonical_form FROM lexemes WHERE id = ?", (lid,)).fetchone()
                                if lex_row:
                                    members.append({
                                        "lexeme_id": lid,
                                        "canonical_form": lex_row["canonical_form"],
                                        "weight": weight,
                                    })
                        
                        result.semantic_overlays.append({
                            "concept_cloud": cloud.to_dict(),
                            "space_id": overlay["space_id"],
                            "center_x": overlay["center_x"],
                            "center_y": overlay["center_y"],
                            "radius": overlay["radius"],
                            "members": members,
                        })
    
    return result


# ============================================================
# Scene Similarity API
# ============================================================

@app.post("/api/scenes/similarity")
async def compute_scene_similarity(scene_a_id: int, scene_b_id: int):
    """Compute weighted Jaccard similarity between two scenes."""
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
        
        r1 = cloud_a.mass if cloud_a else 1.0
        r2 = cloud_b.mass if cloud_b else 1.0
        
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