"""Construction of a bounded, deduplicated working projection."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import BoundedAssociativeWorkspace, Conflict, Evidence, QueryFrame, RetrievalHit, WorkspaceBudget, WorkspaceElement, _id


def build_workspace(
    query_frame: QueryFrame,
    activation_result: Any,
    session_context: Mapping[str, Any] | None = None,
    budget: WorkspaceBudget | Mapping[str, Any] | None = None,
    retrieval_hits: Sequence[RetrievalHit] = (),
) -> BoundedAssociativeWorkspace:
    if not retrieval_hits:
        retrieval_hits = tuple(getattr(activation_result, "hits", ()) or ())
    if isinstance(budget, WorkspaceBudget):
        workspace_budget = budget
    else:
        workspace_budget = WorkspaceBudget(**dict(budget or {}))
    workspace = BoundedAssociativeWorkspace(
        workspace_id=_id("workspace", query_frame.query_id, query_frame.session_id),
        query_id=query_frame.query_id,
        session_id=query_frame.session_id,
        budget=workspace_budget,
        gaps=list(query_frame.gaps),
        constraints=list(query_frame.constraints),
        exclusions=list(query_frame.exclusions),
    )
    active = dict(getattr(activation_result, "activations", {}) or {})
    for index, value in enumerate(query_frame.known_elements[:workspace_budget.max_anchors]):
        workspace.add_element(WorkspaceElement(
            element_id=str(value), element_type="entity", payload={"value": value},
            activation=1.0, workspace_functions=["query_anchor", "known_participant"],
            source_ids=[query_frame.query_id], retrieval_paths=["query_frame"],
            provenance=[{"source_id": query_frame.query_id, "source_type": "query_frame"}],
        ), section="anchors")
    for item in query_frame.inherited_elements:
        value = str(item.get("element") or item.get("value") or "")
        if value:
            workspace.add_element(WorkspaceElement(
                element_id=value, element_type="context", payload=dict(item), activation=0.85,
                workspace_functions=["active_context"], source_ids=[str(item.get("source") or "context")],
                retrieval_paths=["context_inheritance"], provenance=[dict(item)],
            ), section="active_context")
    for hit in retrieval_hits:
        element = WorkspaceElement(
            element_id=hit.element_id, element_type=hit.element_type,
            payload=dict(hit.payload), activation=max(active.get(hit.element_id, 0.0), hit.match_score),
            workspace_functions=["retrieval_hit"], source_ids=[hit.source_id],
            retrieval_paths=list(hit.retrieval_path), provenance=list(hit.provenance),
            conflict_ids=list(hit.conflicts),
        )
        stored_element = workspace.add_element(element, section=hit.element_type)
        evidence = Evidence(
            evidence_id=_id("evidence", query_frame.query_id, hit.hit_id), source_type="observation",
            source_id=hit.source_id, event_id=str(hit.payload.get("event_id") or hit.element_id),
            scene_id=hit.payload.get("scene_id"), strength=hit.match_score,
            supports=(hit.element_id,), retrieval_path=hit.retrieval_path, conflicts=hit.conflicts,
        )
        workspace.add_evidence(evidence)
        if evidence.evidence_id not in stored_element.evidence_ids:
            stored_element.evidence_ids.append(evidence.evidence_id)
        for conflict_id in hit.conflicts:
            if not any(item.conflict_id == conflict_id for item in workspace.conflicts):
                workspace.conflicts.append(Conflict(
                    conflict_id=str(conflict_id), subject_id=hit.element_id,
                    competing_ids=(hit.element_id,), reason="retrieval_conflict",
                    severity=0.7, provenance=hit.provenance,
                ))
        workspace.conflicts = workspace.conflicts[:workspace.budget.max_conflicts]
    workspace.status = "READY"
    workspace.resonance_state["activation_visited"] = len(active)
    return workspace
