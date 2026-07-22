"""Nine-stage hybrid operational-space pipeline."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Mapping

from .activation import spread_activation
from .bees import dispatch_bees
from .contracts import BeeTask, Evidence, WorkspaceBudget, WorkspaceElement, _id
from .query_frame import build_query_frame, inherit_context
from .reasoning import build_candidates, build_configurations, build_hypotheses, compile_answer_structure, run_resonance, should_dispatch_bees
from .retrieval import retrieve_direct
from .workspace import build_workspace


class HybridDialoguePipeline:
    def __init__(self, *, budget: WorkspaceBudget | Mapping[str, Any] | None = None, config: Mapping[str, Any] | None = None) -> None:
        self.budget = budget if isinstance(budget, WorkspaceBudget) else WorkspaceBudget(**dict(budget or {}))
        self.config = dict(config or {})

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
        hits = stage("DIRECT_RETRIEVAL", lambda: [] if frame.unresolved_context else retrieve_direct(frame, indexes, limit=int(self.config.get("retrieval_limit", 128))))
        activation = stage("LOCAL_ACTIVATION", lambda: spread_activation(frame, hits, indexes, self.config.get("activation")))
        workspace = stage("BOUNDED_WORKSPACE", lambda: build_workspace(frame, activation, context, self.budget, hits))
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
                excluded = tuple(workspace.exclusions) + tuple(candidate.element_id for candidate in workspace.candidates if candidate.gap_id == gap.gap_id and candidate.score >= 0.68)
                bee_tasks.append(BeeTask(
                    bee_task_id=f"bee_task_{frame.query_id[-8:]}_{index}", task_type=task_type,
                    gap_id=gap.gap_id, anchors=tuple(gap.known_elements), excluded_elements=excluded,
                    max_steps=int(self.config.get("bee_max_steps", 4)), energy_budget=int(self.config.get("bee_energy_budget", 32)),
                ))
        stages.append({"stage": "BEE_DISPATCH", "duration_ms": 0.0, "result": "DISPATCHED" if bee_tasks else "SKIPPED", "count": len(bee_tasks)})
        bee_results = stage("BEE_EXPANSION", lambda: dispatch_bees(bee_tasks, indexes)) if bee_tasks else []
        if bee_results:
            for result in bee_results:
                if result.status != "FOUND" or not result.result_element_id:
                    continue
                element = workspace.add_element(WorkspaceElement(
                    element_id=result.result_element_id, element_type="entity",
                    payload={"element_id": result.result_element_id, "bee_id": result.bee_id},
                    activation=result.score, workspace_functions=["bee_result"],
                    source_ids=[result.bee_id], retrieval_paths=list(result.path),
                    provenance=list(result.provenance),
                ))
                evidence = Evidence(
                    evidence_id=_id("evidence", result.bee_id, result.result_element_id),
                    source_type="bee_result", source_id=result.bee_id,
                    supports=(result.result_element_id,), strength=result.score,
                    retrieval_path=result.path, conflicts=result.conflicts,
                )
                workspace.add_evidence(evidence)
                if evidence.evidence_id not in element.evidence_ids:
                    element.evidence_ids.append(evidence.evidence_id)
            stage("EVENT_CONFIGURATIONS_AFTER_BEES", lambda: build_configurations(workspace).configurations)
            stage("CANDIDATE_BUILD_AFTER_BEES", lambda: build_candidates(workspace).candidates)
            stage("HYPOTHESIS_BUILD_AFTER_BEES", lambda: build_hypotheses(workspace).hypotheses)
            resonance = stage("RESONANCE_AFTER_BEES", lambda: run_resonance(workspace, self.config.get("resonance")))
        answer = stage("ANSWER_STRUCTURE", lambda: compile_answer_structure(workspace))
        return {
            "query_frame": frame.as_dict(), "retrieval_hits": [item.as_dict() for item in hits],
            "activation": activation.as_dict(), "workspace": workspace.as_dict(),
            "retrieval": {"hits": [item.as_dict() for item in hits]},
            "configurations": [item.as_dict() for item in workspace.configurations],
            "candidates": [item.as_dict() for item in workspace.candidates],
            "hypotheses": [item.as_dict() for item in workspace.hypotheses],
            "bee_dispatch": decision,
            "resonance": resonance, "bee_decision": decision,
            "bee_tasks": [item.as_dict() for item in bee_tasks], "bee_results": [item.as_dict() for item in bee_results],
            "answer": answer.as_dict(), "answer_structure": answer.as_dict(),
            "answer_text": __import__("server.v2.hybrid.reasoning", fromlist=["render_answer"]).render_answer(answer),
            "debug_payload_version": "3.0.0",
            "trace": {"query_id": frame.query_id, "stages": stages, "bee_count": len(bee_results), "status": answer.status},
        }
