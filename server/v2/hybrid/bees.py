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
            type("BeeFrame", (), {"query_id": task.bee_task_id, "explicit_predicate": None, "predicate_hypotheses": (), "known_elements": () if task.bee_mode == "FIELD_BEE" else tuple(task.anchors), "surface_focus": None, "session_id": "", "temporal_scope": None, "negations": ()})(),
            search_space,
            limit=max(1, min(task.energy_budget, 32)),
            )
        hit = next(
            (item for item in hits if item.element_id not in set(task.excluded_elements) and float(item.match_score) > 0.0),
            None,
        )
        if hit is None:
            results.append(BeeResult(
                _id("bee", task.bee_task_id), task.bee_task_id, "NO_RESULT",
                energy_spent=min(task.energy_budget, task.max_steps),
                reason="NO_EVIDENCE",
                bee_mode=task.bee_mode,
                actual_cost=min(task.energy_budget, task.max_steps),
                utility={"useful": False, "new_cloud": False, "new_event": False, "new_candidate": False},
            ))
            continue
        payload = dict(getattr(hit, "payload", {}) or {})
        independent_source_key = next(
            (
                str(item.get("independent_source_key"))
                for item in hit.provenance
                if isinstance(item, dict) and item.get("independent_source_key")
            ),
            None,
        )
        is_field = task.bee_mode == "FIELD_BEE"
        status = "SPATIAL_SUPPORT_FOUND" if is_field else "GRAPH_EVIDENCE_FOUND"
        support_id = str(payload.get("spatial_support_id") or "")
        results.append(BeeResult(
            bee_id=_id("bee", task.bee_task_id), task_id=task.bee_task_id, status=status,
            result_element_id=hit.element_id, result_element_type=hit.element_type,
            path=tuple(["query_anchor", *hit.retrieval_path]),
            score=hit.match_score, energy_spent=min(task.energy_budget, task.max_steps),
            provenance=hit.provenance, conflicts=hit.conflicts,
            reason="field traversal returned spatial support" if is_field else "typed retrieval task returned graph evidence",
            bee_mode=task.bee_mode,
            payload=payload,
            spatial_support_ids=(support_id,) if is_field and support_id else (),
            graph_evidence_ids=() if is_field else (_id("graph-evidence", task.bee_task_id, hit.element_id),),
            independent_source_key=None if is_field else independent_source_key,
            route=tuple(["field_gradient"] if is_field else ["graph_retrieval", *hit.retrieval_path]),
            actual_cost=min(task.energy_budget, task.max_steps),
            utility={
                "score": hit.match_score,
                "mode": task.bee_mode,
                "useful": bool(hit.match_score > 0.0),
                "new_cloud": is_field,
                "new_event": not is_field,
                "new_candidate": False,
            },
        ))
    return results


class BeeDispatcher:
    def dispatch(self, tasks: Sequence[BeeTask], global_space: Any = None) -> list[BeeResult]:
        return dispatch_bees(tasks, global_space)
