"""Nine-stage hybrid operational-space pipeline."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Mapping

from .activation import spread_activation
from .bees import dispatch_bees
from .contracts import BeeTask, GraphEvidence, SpatialSupport, WorkspaceBudget, WorkspaceElement, _id
from .query_frame import build_query_frame, inherit_context
from .reasoning import build_candidates, build_configurations, build_hypotheses, compile_answer_structure, run_resonance, should_dispatch_bees
from .retrieval import retrieve_direct
from .workspace import build_workspace


class HybridDialoguePipeline:
    def __init__(self, *, budget: WorkspaceBudget | Mapping[str, Any] | None = None, config: Mapping[str, Any] | None = None, field_service: Any = None) -> None:
        self.budget = budget if isinstance(budget, WorkspaceBudget) else WorkspaceBudget(**dict(budget or {}))
        self.config = dict(config or {})
        self.field_service = field_service

    def run(
        self,
        text: str,
        *,
        session_context: Mapping[str, Any] | None = None,
        indexes: Any = None,
        analysis: Any = None,
    ) -> dict[str, Any]:
        context = dict(session_context or {})
        stages: list[dict[str, Any]] = []

        def stage(name: str, fn: Any) -> Any:
            started = perf_counter()
            value = fn()
            duration = (perf_counter() - started) * 1000.0
            summary = {"stage": name, "duration_ms": round(duration, 3), "result": "SUCCESS"}
            if hasattr(value, "__len__") and not isinstance(value, (str, bytes, Mapping)):
                summary["count"] = len(value)
            stages.append(summary)
            return value

        frame = stage("QUERY_FRAME_BUILD", lambda: build_query_frame(text, context, session_id=str(context.get("session_id") or ""), analysis=analysis))
        frame = stage("CONTEXT_INHERITANCE", lambda: inherit_context(frame, context))
        field_projection = stage("QUERY_FIELD_PROJECTION", lambda: self.field_service.project_query(frame) if self.field_service is not None else {"anchor_clouds": [], "field_region": {}, "positive_gradients": []})
        field_hits = stage("FIELD_NEIGHBOURHOOD_RETRIEVAL", lambda: self.field_service.neighbourhood(field_projection, limit=int(self.config.get("field_retrieval_limit", 32))) if self.field_service is not None and not frame.unresolved_context else [])
        graph_hits = stage("GRAPH_EVIDENCE_RETRIEVAL", lambda: [] if frame.unresolved_context else retrieve_direct(frame, indexes, limit=int(self.config.get("retrieval_limit", 128))))
        hits = [*field_hits, *graph_hits]
        activation = stage("LOCAL_ACTIVATION", lambda: spread_activation(frame, hits, indexes, self.config.get("activation")))
        workspace = stage("BOUNDED_WORKSPACE", lambda: build_workspace(frame, activation, context, self.budget, hits, field_projection))
        if frame.unresolved_context:
            workspace.status = "UNRESOLVED_CONTEXT"
        stage("EVENT_CONFIGURATIONS", lambda: build_configurations(workspace).configurations)
        stage("CANDIDATE_BUILD", lambda: build_candidates(workspace).candidates)
        stage("HYPOTHESIS_BUILD", lambda: build_hypotheses(workspace).hypotheses)
        resonance = stage("RESONANCE", lambda: run_resonance(workspace, self.config.get("resonance")))
        if frame.unresolved_context:
            workspace.status = "UNRESOLVED_CONTEXT"
            resonance["status"] = workspace.status
            decision = {"dispatch": False, "reasons": ["UNRESOLVED_CONTEXT"], "task_types": [], "bee_count": 0}
        else:
            decision = should_dispatch_bees(workspace, resonance)
        bee_tasks: list[BeeTask] = []
        if decision["dispatch"]:
            for index, gap in enumerate(workspace.gaps[: decision.get("bee_count", 0)]):
                task_type = decision["task_types"][0] if decision.get("task_types") else "FIND_GAP_FILL"
                if task_type == "FIND_GAP_FILL":
                    task_type = {
                        "component": "SEARCH_MORPHOLOGICAL_ANALOGY",
                        "predicate": "SEARCH_SIMILAR_PATTERN",
                        "location": "SEARCH_OTHER_SCENE",
                        "time": "VERIFY_TEMPORAL_COMPATIBILITY",
                    }.get(gap.expected_type, task_type)
                excluded = tuple(
                    str(item.get("element_id") or item.get("entity_id") or item.get("lemma") or item.get("surface") or "")
                    if isinstance(item, Mapping) else str(item)
                    for item in (*workspace.exclusions, *gap.exclusions)
                ) + tuple(candidate.element_id for candidate in workspace.candidates if candidate.gap_id == gap.gap_id and candidate.score >= 0.68)
                bee_tasks.append(BeeTask(
                    bee_task_id=f"bee_task_{frame.query_id[-8:]}_{index}", task_type=task_type,
                    gap_id=gap.gap_id, anchors=tuple(gap.known_elements), excluded_elements=excluded,
                    max_steps=int(self.config.get("bee_max_steps", 4)), energy_budget=int(self.config.get("bee_energy_budget", 32)),
                    bee_mode=("FIELD_BEE" if "NO_SEMANTIC_NEIGHBOURHOOD" in decision.get("reasons", []) or (field_projection.get("anchor_clouds") and not workspace.graph_evidence) else "GRAPH_BEE"),
                ))
        stages.append({"stage": "BEE_DISPATCH", "duration_ms": 0.0, "result": "DISPATCHED" if bee_tasks else "SKIPPED", "count": len(bee_tasks)})
        bee_space = {"records": indexes, "field_records": [item.payload | {"element_id": item.element_id, "element_type": item.element_type, "source_id": item.source_id, "provenance": item.provenance} for item in field_hits]}
        bee_results = stage("BEE_EXPANSION", lambda: dispatch_bees(bee_tasks, bee_space)) if bee_tasks else []
        if bee_results:
            for result in bee_results:
                if result.status != "FOUND" or not result.result_element_id:
                    continue
                element = workspace.add_element(WorkspaceElement(
                    element_id=result.result_element_id, element_type=result.result_element_type or "entity",
                    payload={**dict(result.payload), "element_id": result.result_element_id, "bee_id": result.bee_id},
                    activation=result.score, workspace_functions=["bee_result"],
                    source_ids=[result.bee_id], retrieval_paths=list(result.path),
                    provenance=list(result.provenance),
                ))
                if result.bee_mode == "FIELD_BEE":
                    support_id = (result.spatial_support_ids or (_id("spatial-support", result.bee_id, result.result_element_id),))[0]
                    element.payload["spatial_support_id"] = support_id
                    if not any(item.support_id == support_id for item in workspace.spatial_support):
                        workspace.spatial_support.append(SpatialSupport(
                            support_id=support_id,
                            cloud_id=result.result_element_id,
                            region_id=str(field_projection.get("field_region", {}).get("region_id") or "") or None,
                            score=result.score,
                            retrieval_path=result.route or result.path,
                        ))
                    continue
                evidence = GraphEvidence(
                    evidence_id=_id("evidence", result.bee_id, result.result_element_id),
                    source_type="bee_result", source_id=result.bee_id,
                    supports=(result.result_element_id,), strength=result.score,
                    retrieval_path=result.path, conflicts=result.conflicts,
                    independent_source_key=result.independent_source_key or result.bee_id,
                )
                workspace.add_evidence(evidence)
                if evidence.evidence_id not in element.evidence_ids:
                    element.evidence_ids.append(evidence.evidence_id)
            stage("EVENT_CONFIGURATIONS_AFTER_BEES", lambda: build_configurations(workspace).configurations)
            stage("CANDIDATE_BUILD_AFTER_BEES", lambda: build_candidates(workspace).candidates)
            stage("HYPOTHESIS_BUILD_AFTER_BEES", lambda: build_hypotheses(workspace).hypotheses)
            resonance = stage("RESONANCE_AFTER_BEES", lambda: run_resonance(workspace, self.config.get("resonance")))
        answer = stage("ANSWER_STRUCTURE", lambda: compile_answer_structure(workspace))
        query_graph = analysis.get("query_graph") if isinstance(analysis, Mapping) and isinstance(analysis.get("query_graph"), Mapping) else {}
        return {
            "debug_payload_version": "4.0.0",
            "turn": {"query_id": frame.query_id, "session_id": frame.session_id},
            "query_graph": query_graph,
            "query_frame": frame.as_dict(),
            "query_field_projection": field_projection,
            "retrieval": {"hits": [item.as_dict() for item in hits], "field_hit_count": len(field_hits), "graph_hit_count": len(graph_hits)},
            "activation": activation.as_dict(), "workspace": workspace.as_dict(),
            "configurations": [item.as_dict() for item in workspace.configurations],
            "candidates": [item.as_dict() for item in workspace.candidates],
            "hypotheses": [item.as_dict() for item in workspace.hypotheses],
            "resonance": resonance,
            "bees": {"decision": decision, "tasks": [item.as_dict() for item in bee_tasks], "results": [item.as_dict() for item in bee_results]},
            "answer_structure": answer.as_dict(),
            "answer_text": __import__("server.v2.hybrid.reasoning", fromlist=["render_answer"]).render_answer(answer),
            "errors": [],
            "metrics": {"stages": stages, "bee_count": len(bee_results)},
            "trace": {"query_id": frame.query_id, "stages": stages, "bee_count": len(bee_results), "status": answer.status},
        }
