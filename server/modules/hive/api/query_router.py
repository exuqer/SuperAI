"""Public query-scene API kept separate from legacy hive routing."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from server.modules.hive.application.services import HiveService
from server.modules.hive.infrastructure.repository import HiveRepository


router = APIRouter(prefix="/api", tags=["query-scenes"])
_frames: dict[str, dict[str, Any]] = {}


def service() -> HiveService:
    return HiveService(HiveRepository())


class TextRequest(BaseModel):
    text: str = Field(min_length=1)


class ActivateRequest(BaseModel):
    query_frame_id: str
    max_cells: int = Field(default=24, ge=1, le=128)


class VibrationRequest(BaseModel):
    hive_id: str
    steps: int = Field(default=3, ge=1, le=32)
    config: dict[str, Any] = Field(default_factory=dict)


class UnknownTokenSearchRequest(BaseModel):
    hive_id: str
    surface: str = Field(min_length=1)
    token_index: int = Field(default=0, ge=0)
    query_role: str = ""
    query_scene_id: str = ""


class UnknownTokenSearchReference(BaseModel):
    hive_id: str
    search_id: str


class ResonanceRunRequest(BaseModel):
    text: str = Field(min_length=1)
    scope: str = "LOCAL_THEN_GLOBAL"


class ResonanceSessionRequest(BaseModel):
    input: str = Field(min_length=1)
    temperature: float = Field(default=.25, ge=0, le=1)
    max_ticks: int = Field(default=8, ge=1, le=64)
    use_global_memory: bool = True
    save_snapshots: bool = True


class ResonanceImportRequest(BaseModel):
    match_id: str = Field(min_length=1)
    include_scenes: bool = False


@router.post("/resonance/classify")
async def classify_resonance(request: TextRequest) -> dict[str, Any]:
    return service().classify_resonance(request.text)


@router.get("/resonance/{session_id}")
async def get_resonance_session(session_id: str) -> dict[str, Any]:
    return service().facade.hive_resonance.get(session_id)


@router.post("/resonance/{session_id}/tick")
async def tick_resonance_session(session_id: str) -> dict[str, Any]:
    return service().facade.hive_resonance.step(session_id)


@router.post("/resonance/{session_id}/run")
async def run_resonance_session(session_id: str) -> dict[str, Any]:
    return service().facade.hive_resonance.run(session_id)


@router.get("/resonance/{session_id}/snapshots")
async def resonance_session_snapshots(session_id: str) -> dict[str, Any]:
    return {"snapshots": service().resonance_snapshots(session_id)}


@router.post("/resonance/{session_id}/stop")
async def stop_resonance_session(session_id: str) -> dict[str, Any]:
    return service().resonance_stop(session_id)


@router.post("/hive/{hive_id}/resonance")
async def create_resonance(hive_id: str, request: ResonanceRunRequest) -> dict[str, Any]:
    return {"probe": service().resonance_create(hive_id, request.text, request.scope)}


@router.post("/hive/{hive_id}/resonance/{probe_id}/step")
async def step_resonance(hive_id: str, probe_id: str) -> dict[str, Any]:
    return {"probe": service().resonance_step(hive_id, probe_id)}


@router.post("/hive/{hive_id}/resonance/{probe_id}/run")
async def run_resonance(hive_id: str, probe_id: str) -> dict[str, Any]:
    return service().resonance_run(hive_id, probe_id)


@router.get("/hive/{hive_id}/resonance/{probe_id}")
async def get_resonance(hive_id: str, probe_id: str) -> dict[str, Any]:
    return {"probe": service().resonance_get(hive_id, probe_id)}


@router.post("/hive/{hive_id}/resonance/{probe_id}/import")
async def import_resonance(hive_id: str, probe_id: str, request: ResonanceImportRequest) -> dict[str, Any]:
    return service().resonance_import(hive_id, probe_id, request.match_id, request.include_scenes)


@router.get("/hive/{hive_id}/resonance/{probe_id}/related-scenes")
async def related_resonance_scenes(hive_id: str, probe_id: str, match_id: str = "") -> dict[str, Any]:
    return service().resonance_related_scenes(hive_id, probe_id, match_id)


@router.post("/query/parse")
async def parse_query(request: TextRequest) -> dict[str, Any]:
    parsed = service().parse_query(request.text)
    _frames[parsed["query_frame"]["id"]] = parsed
    return parsed


@router.post("/hive/activate")
async def activate_hive(request: ActivateRequest) -> dict[str, Any]:
    parsed = _frames.get(request.query_frame_id)
    if not parsed:
        raise HTTPException(status_code=404, detail="query frame not found")
    hive_service = service()
    hive = hive_service.create(request.max_cells)["hive"]
    active = hive_service.activate_query(hive["id"], parsed["query_frame"]["source_text"])
    return {
        "hive": active,
        "memory_scenes": active["memory_scenes"],
        "candidates": active["candidates"],
    }


@router.post("/hive/vibrate/step")
async def vibrate_step(request: VibrationRequest) -> dict[str, Any]:
    return service().vibration_step(request.hive_id, request.config)


@router.post("/hive/vibrate/run")
async def vibrate_run(request: VibrationRequest) -> dict[str, Any]:
    return service().vibration_run(request.hive_id, request.steps, request.config)


@router.post("/hive/vibrate/stop")
async def vibrate_stop(request: VibrationRequest) -> dict[str, Any]:
    return service().vibration_stop(request.hive_id)


@router.post("/hive/unknown-token/search")
async def start_unknown_token_search(request: UnknownTokenSearchRequest) -> dict[str, Any]:
    search = service().unknown_search_start(request.hive_id, request.surface, request.token_index, request.query_role, request.query_scene_id)
    return {"search": search, "current_mode": search["current_mode"], "candidates": []}


@router.post("/hive/unknown-token/search/step")
async def step_unknown_token_search(request: UnknownTokenSearchReference) -> dict[str, Any]:
    return service().unknown_search_step(request.hive_id, request.search_id)


@router.post("/hive/unknown-token/search/run")
async def run_unknown_token_search(request: UnknownTokenSearchReference) -> dict[str, Any]:
    return service().unknown_search_run(request.hive_id, request.search_id)


@router.post("/hive/unknown-token/vibrate/step")
async def vibrate_unknown_token_search(request: UnknownTokenSearchReference) -> dict[str, Any]:
    return service().unknown_search_vibrate(request.hive_id, request.search_id)


@router.get("/hive/{hive_id}/unknown-search/{search_id}")
async def unknown_token_search(hive_id: str, search_id: str) -> dict[str, Any]:
    return service().unknown_search_get(hive_id, search_id)


@router.get("/hive/{hive_id}/unknown-search/{search_id}/evidence")
async def unknown_token_search_evidence(hive_id: str, search_id: str) -> dict[str, Any]:
    return {"evidence": service().unknown_search_evidence(hive_id, search_id)}


@router.get("/hive/{hive_id}/unknown-search/{search_id}/routes")
async def unknown_token_search_routes(hive_id: str, search_id: str) -> dict[str, Any]:
    return {"routes": service().unknown_search_routes(hive_id, search_id)}


@router.post("/hive/unknown-token/confirm")
async def confirm_unknown_token_search(request: UnknownTokenSearchReference) -> dict[str, Any]:
    return service().unknown_search_confirm(request.hive_id, request.search_id)


@router.get("/hive/{hive_id}")
async def get_hive(hive_id: str) -> dict[str, Any]:
    return service().query_working_state(hive_id)


@router.get("/hive/{hive_id}/json")
async def get_hive_json(hive_id: str) -> dict[str, Any]:
    return {"hive": service().query_working_state(hive_id)}


@router.get("/hive/{hive_id}/analytics")
async def get_hive_analytics(hive_id: str) -> dict[str, Any]:
    hive = service().query_working_state(hive_id)
    scenes = hive["memory_scenes"]
    best = max(scenes, key=lambda item: item["scores"]["total_score"], default=None)
    return {
        "activation": hive["energy"], "retention": max((item["scores"].get("retention", 0) for item in hive["candidates"]), default=0),
        "object_match": best["scores"]["object_match"] if best else 0, "action_match": best["scores"]["action_match"] if best else 0,
        "requested_role_match": best["scores"]["requested_role_match"] if best else 0,
        "grammar_match": best["scores"]["grammar_match"] if best else 0,
        "answer_confidence": hive["answer"]["confidence"], "result_type": hive["result_type"],
    }


@router.get("/hive/{hive_id}/history")
async def get_hive_history(hive_id: str) -> dict[str, Any]:
    return {"history": service().query_working_state(hive_id)["vibration"]["history"]}


@router.get("/hive/{hive_id}/projection/scene")
async def scene_projection(hive_id: str) -> dict[str, Any]:
    hive = service().query_working_state(hive_id)
    return {"query_scene": hive["query_scene"], "memory_scenes": hive["memory_scenes"]}


@router.get("/hive/{hive_id}/projection/lexeme/{lexeme_id}")
async def lexeme_projection(hive_id: str, lexeme_id: str) -> dict[str, Any]:
    hive = service().query_working_state(hive_id)
    lemma = lexeme_id.removeprefix("concept-")
    links = []
    for scene in hive["memory_scenes"]:
        for role, value in scene["roles"].items():
            if value.get("lemma") == lemma:
                links.append({"scene_id": scene["id"], "role": role, "relation": "scene_role"})
    return {"lemma": lemma, "links": links, "semantic_neighbors": []}


@router.get("/hive/{hive_id}/projection/morphology/{lexeme_id}")
async def morphology_projection(hive_id: str, lexeme_id: str) -> dict[str, Any]:
    hive = service().query_working_state(hive_id)
    lemma = lexeme_id.removeprefix("concept-")
    values = [value for scene in hive["memory_scenes"] for value in scene["roles"].values() if value.get("lemma") == lemma]
    return {"lemma": lemma, "forms": [{"surface": value["surface"], "features": value.get("grammatical_features", {})} for value in values], "relation": "lexeme_form"}
