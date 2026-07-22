"""Construction of a bounded, deduplicated working projection."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import BoundedAssociativeWorkspace, Conflict, EnumerationPolicy, EnumerationState, GraphEvidence, QueryFrame, RetrievalHit, SpatialSupport, WorkspaceBudget, WorkspaceElement, _id


def query_reference_id(value: object) -> str:
    """Return a stable identity for a structured query reference."""
    if isinstance(value, Mapping):
        return str(
            value.get("concept_id")
            or value.get("entity_id")
            or value.get("node_id")
            or value.get("mention_id")
            or value.get("lemma")
            or value.get("surface")
            or ""
        )
    return str(value or "")


def spatial_support_identity(
    query_id: str,
    cloud_id: str,
    field_revision: int,
    region_id: str | None,
) -> str:
    """Identity of a spatial observation, independent of the retrieval route."""
    return _id(
        "spatial-support",
        query_id,
        int(field_revision or 0),
        cloud_id,
        region_id or "",
    )


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
    raw_text = str(query_frame.raw_text or "").casefold()
    if any(marker in raw_text for marker in ("перечисли", "все", "всё", "назови всё", "полный список", "какие объекты")):
        enumeration_policy = EnumerationPolicy.RETURN_ALL.value
    elif any(token.isdigit() for token in raw_text.split()):
        enumeration_policy = EnumerationPolicy.TOP_K.value
    elif query_frame.query_type in {"continuation_question", "continuation_relation_question"}:
        enumeration_policy = EnumerationPolicy.FIRST_THEN_CONTINUE.value
    else:
        enumeration_policy = EnumerationPolicy.FIRST_THEN_CONTINUE.value
    inherited_enumeration = (session_context or {}).get("enumeration_state")
    enumeration_state = None
    if isinstance(inherited_enumeration, EnumerationState):
        enumeration_state = inherited_enumeration
    elif isinstance(inherited_enumeration, Mapping) and inherited_enumeration.get("enumeration_id"):
        try:
            enumeration_state = EnumerationState(**dict(inherited_enumeration))
        except TypeError:
            enumeration_state = None
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
        explicit_predicate=query_frame.explicit_predicate,
        predicate_hypotheses=list(query_frame.predicate_hypotheses),
        field_region=dict((field_projection or {}).get("field_region") or {}),
        local_gradients=list((field_projection or {}).get("positive_gradients") or ()),
        enumeration_policy=enumeration_policy,
        enumeration_state=enumeration_state,
    )
    spatial_by_hit: dict[str, SpatialSupport] = {}
    active = dict(getattr(activation_result, "activations", {}) or {})
    for index, value in enumerate(query_frame.known_elements[:workspace_budget.max_anchors]):
        reference_id = query_reference_id(value)
        if not reference_id:
            continue
        reference_payload = dict(value) if isinstance(value, Mapping) else {"value": value}
        workspace.add_element(WorkspaceElement(
            element_id=reference_id, element_type="entity", payload={"reference": reference_payload, **reference_payload},
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
            cloud_id = str(hit.payload.get("cloud_id") or hit.element_id)
            field_revision = int(hit.payload.get("field_revision") or 0)
            region_id = str((field_projection or {}).get("field_region", {}).get("region_id") or f"region:{query_frame.query_id}")
            support = SpatialSupport(
                support_id=spatial_support_identity(
                    query_frame.query_id, cloud_id, field_revision, region_id
                ),
                cloud_id=cloud_id,
                region_id=region_id,
                score=hit.match_score,
                relation_alignment=float(next((item.get("weight") for item in (field_projection or {}).get("relation_projections") or () if isinstance(item, Mapping)), 0.0) or 0.0),
                field_revision=field_revision,
                retrieval_path=hit.retrieval_path,
            )
            spatial_by_hit[hit.hit_id] = support
            existing_support = next((item for item in workspace.spatial_support if item.support_id == support.support_id), None)
            if existing_support is None:
                workspace.spatial_support.append(support)
            else:
                support = existing_support
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
