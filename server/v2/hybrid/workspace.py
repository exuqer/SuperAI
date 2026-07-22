"""Construction of a bounded, deduplicated working projection."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import BoundedAssociativeWorkspace, Conflict, GraphEvidence, QueryFrame, RetrievalHit, SpatialSupport, WorkspaceBudget, WorkspaceElement, _id


def build_workspace(
    query_frame: QueryFrame,
    activation_result: Any,
    session_context: Mapping[str, Any] | None = None,
    budget: WorkspaceBudget | Mapping[str, Any] | None = None,
    retrieval_hits: Sequence[RetrievalHit] = (),
    field_projection: Mapping[str, Any] | None = None,
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
        query_type=query_frame.query_type,
        budget=workspace_budget,
        gaps=list(query_frame.gaps),
        constraints=list(query_frame.constraints),
        exclusions=list(query_frame.exclusions),
        temporal_scope=query_frame.temporal_scope,
        field_region=dict((field_projection or {}).get("field_region") or {}),
        local_gradients=list((field_projection or {}).get("positive_gradients") or ()),
    )
    spatial_by_hit: dict[str, SpatialSupport] = {}
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
        if hit.element_type == "cloud" or hit.origin == "FIELD":
            support = SpatialSupport(
                support_id=_id("spatial-support", query_frame.query_id, hit.hit_id),
                cloud_id=str(hit.payload.get("cloud_id") or hit.element_id),
                region_id=str((field_projection or {}).get("field_region", {}).get("field_revision") or "") or None,
                score=hit.match_score,
                relation_alignment=float(next((item.get("weight") for item in (field_projection or {}).get("relation_projections") or () if isinstance(item, Mapping)), 0.0) or 0.0),
                field_revision=int(hit.payload.get("field_revision") or 0),
                retrieval_path=hit.retrieval_path,
            )
            spatial_by_hit[hit.hit_id] = support
            workspace.spatial_support.append(support)
            stored_element.payload["spatial_support_id"] = support.support_id
            stored_element.payload["spatial_support"] = True
            continue
        evidence = GraphEvidence(
            evidence_id=_id("evidence", query_frame.query_id, hit.hit_id), source_type=hit.origin,
            source_id=hit.source_id, event_id=str(hit.payload.get("event_id") or hit.element_id),
            scene_id=hit.payload.get("scene_id"), strength=hit.match_score,
            supports=(hit.element_id,), retrieval_path=hit.retrieval_path, conflicts=hit.conflicts,
            independent_source_key=next((str(item.get("independent_source_key")) for item in hit.provenance if isinstance(item, Mapping) and item.get("independent_source_key")), None),
        )
        workspace.add_evidence(evidence)
        workspace.graph_evidence.append(evidence)
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
