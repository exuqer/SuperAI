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
        hits = retrieve_direct(
            type("BeeFrame", (), {"query_id": task.bee_task_id, "explicit_predicate": None, "known_elements": tuple(task.anchors), "surface_focus": None, "session_id": "", "temporal_scope": None, "negations": ()})(),
            global_space,
            limit=max(1, min(task.energy_budget, 32)),
        )
        hit = next((item for item in hits if item.element_id not in set(task.excluded_elements)), None)
        if hit is None:
            results.append(BeeResult(_id("bee", task.bee_task_id), task.bee_task_id, "NOT_FOUND", energy_spent=min(task.energy_budget, task.max_steps), reason="NO_EVIDENCE"))
            continue
        results.append(BeeResult(
            bee_id=_id("bee", task.bee_task_id), task_id=task.bee_task_id, status="FOUND",
            result_element_id=hit.element_id, path=tuple(["query_anchor", *hit.retrieval_path]),
            score=hit.match_score, energy_spent=min(task.energy_budget, task.max_steps),
            provenance=hit.provenance, conflicts=hit.conflicts,
            reason="typed retrieval task returned an evidence-bearing hit",
        ))
    return results


class BeeDispatcher:
    def dispatch(self, tasks: Sequence[BeeTask], global_space: Any = None) -> list[BeeResult]:
        return dispatch_bees(tasks, global_space)
