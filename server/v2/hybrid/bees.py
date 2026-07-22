"""Conditional, typed bee expansion over the same deterministic indexes."""

from __future__ import annotations

from typing import Any, Sequence

from .contracts import BeeResult, BeeTask, _id
from .retrieval import retrieve_direct

VALID_TASKS = {
    "FIND_GAP_FILL", "FIND_ALTERNATIVE_GAP_FILL", "FIND_SUPPORTING_EVIDENCE",
    "FIND_CONTRADICTING_EVIDENCE", "RESOLVE_REFERENCE", "EXPAND_EVENT_PATH",
    "SEARCH_SIMILAR_PATTERN", "SEARCH_MORPHOLOGICAL_ANALOGY", "SEARCH_OTHER_SCENE",
    "VERIFY_TEMPORAL_COMPATIBILITY",
}


def dispatch_bees(tasks: Sequence[BeeTask], global_space: Any = None) -> list[BeeResult]:
    results: list[BeeResult] = []
    for task in sorted(tasks, key=lambda item: item.bee_task_id):
        if task.task_type not in VALID_TASKS:
            results.append(BeeResult(_id("bee", task.bee_task_id), task.bee_task_id, "REJECTED", reason="UNKNOWN_TASK_TYPE"))
            continue
        # A bee receives a constrained query, not a global open-ended search.
        search_space = global_space
        if task.bee_mode == "FIELD_BEE":
            records = list(global_space.get("field_records") or ()) if isinstance(global_space, dict) else []
            hits = []
            for record in records:
                if str(record.get("element_id") or record.get("cloud_id") or "") in set(task.excluded_elements):
                    continue
                hits.append(type("FieldHit", (), {
                    "element_id": str(record.get("element_id") or record.get("cloud_id") or ""),
                    "element_type": "cloud", "match_score": float(record.get("match_score") or record.get("activation") or 0.0),
                    "retrieval_path": ("gradient_traversal", "halo_intersection", str(record.get("element_id") or "")),
                    "provenance": tuple(record.get("provenance") or ()), "conflicts": tuple(record.get("conflicts") or ()),
                    "payload": dict(record), "origin": "FIELD",
                })())
            hits.sort(key=lambda item: (-item.match_score, item.element_id))
            hits = hits[:max(1, min(task.energy_budget, 32))]
        else:
            hits = retrieve_direct(
            type("BeeFrame", (), {"query_id": task.bee_task_id, "explicit_predicate": None, "known_elements": () if task.bee_mode == "FIELD_BEE" else tuple(task.anchors), "surface_focus": None, "session_id": "", "temporal_scope": None, "negations": ()})(),
            search_space,
            limit=max(1, min(task.energy_budget, 32)),
            )
        hit = next((item for item in hits if item.element_id not in set(task.excluded_elements)), None)
        if hit is None:
            results.append(BeeResult(_id("bee", task.bee_task_id), task.bee_task_id, "NOT_FOUND", energy_spent=min(task.energy_budget, task.max_steps), reason="NO_EVIDENCE"))
            continue
        results.append(BeeResult(
            bee_id=_id("bee", task.bee_task_id), task_id=task.bee_task_id, status="FOUND",
            result_element_id=hit.element_id, result_element_type=hit.element_type,
            path=tuple(["query_anchor", *hit.retrieval_path]),
            score=hit.match_score, energy_spent=min(task.energy_budget, task.max_steps),
            provenance=hit.provenance, conflicts=hit.conflicts,
            reason="field traversal returned spatial support" if task.bee_mode == "FIELD_BEE" else "typed retrieval task returned an evidence-bearing hit", bee_mode=task.bee_mode,
            payload=dict(getattr(hit, "payload", {}) or {}),
            spatial_support_ids=(_id("spatial-support", task.bee_task_id, hit.element_id),) if task.bee_mode == "FIELD_BEE" else (),
            graph_evidence_ids=() if task.bee_mode == "FIELD_BEE" else (_id("graph-evidence", task.bee_task_id, hit.element_id),),
            route=tuple(["field_gradient"] if task.bee_mode == "FIELD_BEE" else ["graph_retrieval", *hit.retrieval_path]),
            actual_cost=min(task.energy_budget, task.max_steps), utility={"score": hit.match_score, "mode": task.bee_mode},
        ))
    return results


class BeeDispatcher:
    def dispatch(self, tasks: Sequence[BeeTask], global_space: Any = None) -> list[BeeResult]:
        return dispatch_bees(tasks, global_space)
