"""Hive API router."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, Query, Response

from server.core.exceptions import ConflictError
from server.modules.hive.application.services import HiveService
from server.modules.hive.infrastructure.repository import HiveRepository
from server.modules.hive.api.dto import (
    HiveCreateRequest,
    HiveAnalyticsResponse,
    HiveQueryRequest,
    HiveReasoningRequest,
    HiveExpandRequest,
    HiveGenerateRequest,
    HiveValidateSurfaceRequest,
    HiveVibrationRequest,
    LexicalCandidatesRequest,
    ResonanceRequest,
    ResonanceImportRequest,
    ResonanceConceptImportRequest,
)

router = APIRouter(prefix="/api/v2/hives", tags=["hives"])


def get_hive_service() -> HiveService:
    """Dependency for HiveService."""
    return HiveService(HiveRepository())


@router.post("/resonance/classify")
async def classify_resonance(request: HiveQueryRequest, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.classify_resonance(request.text)


@router.post("/query/parse")
async def parse_query_scene(request: HiveQueryRequest, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.parse_query(request.text)


@router.post("")
async def create_hive(
    request: HiveCreateRequest,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, Any]:
    """Create a new hive."""
    return service.create(request.max_cells, request.conversation_id)


@router.get("/{hive_id}")
async def get_hive(
    hive_id: str,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, Any]:
    """Get hive by ID."""
    return service.get_hive(hive_id)


@router.post("/{hive_id}/query/activate")
async def activate_query_scene(hive_id: str, request: HiveQueryRequest, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.activate_query(hive_id, request.text, request.resolved_mode or "NEW_QUERY")


@router.post("/{hive_id}/vibrate/step")
async def vibrate_query_scene_step(hive_id: str, request: HiveVibrationRequest, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.vibration_step(hive_id, request.config)


@router.post("/{hive_id}/vibrate/run")
async def vibrate_query_scene_run(hive_id: str, request: HiveVibrationRequest, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.vibration_run(hive_id, request.steps, request.config)


@router.post("/{hive_id}/vibrate/stop")
async def stop_query_scene_vibration(hive_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.vibration_stop(hive_id)


@router.get("/{hive_id}/dynamics")
async def hive_dynamics(hive_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return {"dynamics": service.dynamics_state(hive_id)}


@router.get("/{hive_id}/dynamics/history")
async def hive_dynamics_history(hive_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return {"history": service.dynamics_history(hive_id)}


@router.post("/{hive_id}/dynamics/step")
async def hive_dynamics_step(hive_id: str, request: HiveVibrationRequest, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return {"dynamics": service.dynamics_step(hive_id, request.config)}


@router.post("/{hive_id}/dynamics/reset")
async def hive_dynamics_reset(hive_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return {"dynamics": service.dynamics_reset(hive_id)}


@router.post("/{hive_id}/dynamics/replay")
async def hive_dynamics_replay(hive_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return {"snapshots": service.dynamics_history(hive_id)}


@router.get("/{hive_id}/dynamics/node/{cell_id}")
async def hive_dynamics_node(hive_id: str, cell_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return {"node": service.dynamics_node(hive_id, cell_id)}


@router.get("/{hive_id}/dynamics/evictions")
async def hive_dynamics_evictions(hive_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return {"evictions": service.dynamics_evictions(hive_id)}


@router.get("/{hive_id}/query-state")
async def query_scene_state(hive_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.query_working_state(hive_id)


@router.get("/{hive_id}/json")
async def query_scene_json(hive_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return {"hive": service.query_working_state(hive_id)}


@router.get("/{hive_id}/history")
async def query_scene_history(hive_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return {"history": service.query_working_state(hive_id)["vibration"]["history"]}


@router.get("/{hive_id}/hierarchy")
async def hive_hierarchy(hive_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.hierarchy(hive_id)


@router.get("/{hive_id}/views/root")
async def root_hive_view(hive_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.view(hive_id)


@router.get("/{hive_id}/views/{view_id}")
async def hive_view(hive_id: str, view_id: int, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.view(hive_id, view_id)


@router.post("/{hive_id}/cells/{cell_id}/expand")
async def expand_hive_cell(hive_id: str, cell_id: str, request: HiveExpandRequest,
                           service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.expand(hive_id, cell_id, request.target_level, request.reason, request.max_candidates)


@router.post("/{hive_id}/subspaces/{subspace_id}/collapse")
async def collapse_hive_subspace(hive_id: str, subspace_id: int,
                                 service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.collapse(hive_id, subspace_id)


@router.get("/{hive_id}/generation-candidates")
async def generation_candidates(hive_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return {"candidates": service.candidates(hive_id)}


@router.get("/{hive_id}/generation-candidates/{candidate_id}")
async def generation_candidate(hive_id: str, candidate_id: int,
                               service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    candidates = service.candidates(hive_id, candidate_id)
    return candidates[0]


@router.post("/{hive_id}/generation-candidates/{candidate_id}/select")
async def select_generation_candidate(hive_id: str, candidate_id: int,
                                      service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.select_candidate(hive_id, candidate_id)


@router.post("/{hive_id}/generate")
async def generate_surface(hive_id: str, request: HiveGenerateRequest,
                           service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.generate(hive_id, request.sentence_plan)


@router.post("/{hive_id}/validate-surface")
async def validate_surface(hive_id: str, request: HiveValidateSurfaceRequest,
                           service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.validate_surface(hive_id, request.surface)


@router.post("/{hive_id}/query/preview")
async def preview_hive(
    hive_id: str,
    request: HiveQueryRequest,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, Any]:
    """Preview routing decision without modifying state."""
    return service.preview(hive_id, request.text)


@router.post("/{hive_id}/query")
async def query_hive(
    hive_id: str,
    request: HiveQueryRequest,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, Any]:
    """Process a query against the hive."""
    return service.query(hive_id, request.text, request.resolved_mode, request.resonance_scope)


@router.post("/{hive_id}/lexical-candidates")
async def lexical_candidates(hive_id: str, request: LexicalCandidatesRequest, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.lexical_candidates(hive_id, request.text, request.use_global_memory)


@router.post("/{hive_id}/resonance")
async def create_resonance_session(hive_id: str, request: ResonanceRequest, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    session = service.resonance_create(
        hive_id, request.input, temperature=request.temperature, max_ticks=request.max_ticks,
        use_global_memory=request.use_global_memory, save_snapshots=request.save_snapshots,
        config=request.config,
    )
    return {"session_id": session["id"], "status": session["status"], "session": session}


@router.post("/{hive_id}/resonance/{probe_id}/step")
async def step_resonance_probe(hive_id: str, probe_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.resonance_step(hive_id, probe_id)


@router.post("/{hive_id}/resonance/{probe_id}/run")
async def run_resonance_probe(hive_id: str, probe_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.resonance_run(hive_id, probe_id)


@router.post("/{hive_id}/resonance/{probe_id}/stop")
async def stop_resonance_session(hive_id: str, probe_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.resonance_stop(probe_id)


@router.get("/{hive_id}/resonance/{probe_id}/snapshots")
async def resonance_snapshots(hive_id: str, probe_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return {"snapshots": service.resonance_snapshots(probe_id)}


@router.get("/{hive_id}/resonance/{probe_id}")
async def get_resonance_probe(hive_id: str, probe_id: str, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.resonance_get(hive_id, probe_id)


@router.post("/{hive_id}/resonance/{probe_id}/import")
async def import_resonance_match(hive_id: str, probe_id: str, request: ResonanceImportRequest, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.resonance_import(hive_id, probe_id, request.match_id, request.include_scenes)


@router.post("/{hive_id}/import-concept")
async def import_resonance_concept(hive_id: str, request: ResonanceConceptImportRequest, service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.import_resonance_concept(request.session_id, request.concept_id)


@router.get("/{hive_id}/resonance/{probe_id}/related-scenes")
async def resonance_related_scenes(hive_id: str, probe_id: str, match_id: str = "", service: HiveService = Depends(get_hive_service)) -> dict[str, Any]:
    return service.resonance_related_scenes(hive_id, probe_id, match_id)


@router.get("/{hive_id}/resonance-events")
async def hive_events(
    hive_id: str,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, list[dict[str, Any]]]:
    """Get resonance events for hive."""
    return {"events": service.events(hive_id)}


@router.get("/{hive_id}/search-decisions")
async def hive_decisions(
    hive_id: str,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, list[dict[str, Any]]]:
    """Get search decisions for hive."""
    return {"decisions": service.decisions(hive_id)}


@router.get("/{hive_id}/cells/{cell_id}/matches")
async def hive_matches(
    hive_id: str,
    cell_id: str,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, list[dict[str, Any]]]:
    """Get cell matches for hive."""
    return {"matches": service.matches(hive_id, cell_id)}


@router.post("/{hive_id}/reasoning")
async def reason_hive(
    hive_id: str,
    request: HiveReasoningRequest,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, Any]:
    """Run reasoning on hive."""
    return service.reason(hive_id, request.text, request.config)


@router.post("/{hive_id}/reasoning/step")
async def reason_hive_step(
    hive_id: str,
    request: HiveReasoningRequest,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, Any]:
    """Run single reasoning step on hive."""
    config = dict(request.config)
    config["reasoning_steps"] = 1
    return service.reason(hive_id, request.text, config)


@router.post("/{hive_id}/reasoning/stop")
async def stop_hive_reasoning(
    hive_id: str,
    service: HiveService = Depends(get_hive_service),
) -> Response:
    """Stop hive reasoning (not supported for synchronous runs)."""
    raise ConflictError("synchronous reasoning runs cannot be stopped")


@router.get("/{hive_id}/reasoning-runs")
async def reasoning_runs(
    hive_id: str,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, list[dict[str, Any]]]:
    """Get reasoning runs for hive."""
    return {"runs": service.runs(hive_id)}


@router.get("/{hive_id}/analytics", response_model=HiveAnalyticsResponse)
async def hive_analytics(
    hive_id: str,
    run_id: str = Query(default=""),
    compare_run_id: str = Query(default=""),
    service: HiveService = Depends(get_hive_service),
) -> dict[str, Any]:
    """Return read-only analysis of one or two persisted reasoning runs."""
    return service.analytics(hive_id, run_id or None, compare_run_id or None)


@router.get("/{hive_id}/reasoning-runs/{run_id}/snapshots")
async def reasoning_snapshots(
    hive_id: str,
    run_id: str,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, list[dict[str, Any]]]:
    """Get snapshots for reasoning run."""
    return {"snapshots": service.snapshots(hive_id, run_id)}


@router.post("/{hive_id}/reasoning-runs/{run_id}/snapshots/{step}/restore")
async def restore_reasoning_snapshot(
    hive_id: str,
    run_id: str,
    step: int,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, Any]:
    """Restore hive to reasoning snapshot."""
    return service.restore(hive_id, run_id, step)


@router.get("/{hive_id}/reasoning-runs/{run_id}/diff")
async def reasoning_diff(
    hive_id: str,
    run_id: str,
    from_step: int,
    to_step: int,
    service: HiveService = Depends(get_hive_service),
) -> dict[str, Any]:
    """Get diff between reasoning steps."""
    return service.diff(run_id, from_step, to_step)


@router.get("/{hive_id}/reasoning/export")
async def reasoning_export(
    hive_id: str,
    mode: str = Query(default="current"),
    run_id: str = Query(default=""),
    step: Optional[int] = Query(default=None),
    detail: str = Query(default="full"),
    service: HiveService = Depends(get_hive_service),
) -> dict[str, Any]:
    """Export hive reasoning data."""
    return service.export(hive_id, mode, run_id or None, step, detail)


@router.get("/{hive_id}/reasoning/export/download")
async def reasoning_export_download(
    hive_id: str,
    mode: str = Query(default="current"),
    run_id: str = Query(default=""),
    step: Optional[int] = Query(default=None),
    detail: str = Query(default="full"),
    service: HiveService = Depends(get_hive_service),
) -> Response:
    """Download hive reasoning export as JSON file."""
    import json
    payload = service.export(hive_id, mode, run_id or None, step, detail)
    return Response(
        content=json.dumps(payload, ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="hive-{hive_id}-{mode}.json"'},
    )
