"""Bounded local activation around query anchors and retrieval hits."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import ActivationResult, QueryFrame, RetrievalHit, clamp


def spread_activation(
    query_frame: QueryFrame,
    retrieval_hits: Sequence[RetrievalHit],
    global_space: Any = None,
    config: Mapping[str, Any] | None = None,
) -> ActivationResult:
    cfg = {
        "activation_steps": 2, "propagation_decay": 0.62,
        "minimum_activation": 0.12, "max_activated_elements": 256,
        "max_neighbors_per_element": 32,
    }
    cfg.update(dict(config or {}))
    activations: dict[str, float] = {}
    paths: dict[str, list[str]] = {}
    for hit in retrieval_hits:
        activations[hit.element_id] = max(activations.get(hit.element_id, 0.0), clamp(hit.match_score))
        paths[hit.element_id] = list(hit.retrieval_path or ("query_anchor", hit.element_id))
    records = {}
    if isinstance(global_space, Mapping):
        records = global_space.get("elements") or global_space.get("records") or {}
    elif global_space is not None:
        records = global_space
    if isinstance(records, Mapping):
        records = [{"element_id": key, **value} for key, value in records.items() if isinstance(value, Mapping)]
    by_id = {str(item.get("element_id") or item.get("id")): item for item in records or () if isinstance(item, Mapping)}
    for _ in range(max(0, int(cfg["activation_steps"]))):
        additions: dict[str, float] = {}
        for element_id, activation in sorted(activations.items()):
            record = by_id.get(element_id) or {}
            neighbors = record.get("neighbors") or ()
            for neighbor in list(neighbors)[: int(cfg["max_neighbors_per_element"])]:
                if isinstance(neighbor, Mapping):
                    neighbor_id = str(neighbor.get("element_id") or neighbor.get("id") or "")
                    weight = float(neighbor.get("weight") or neighbor.get("transition_weight") or 0.0)
                else:
                    neighbor_id, weight = str(neighbor), 0.25
                if not neighbor_id:
                    continue
                value = activation * float(cfg["propagation_decay"]) * max(0.0, min(1.0, weight))
                if value >= float(cfg["minimum_activation"]):
                    additions[neighbor_id] = max(additions.get(neighbor_id, 0.0), value)
                    paths.setdefault(neighbor_id, paths.get(element_id, []) + [neighbor_id])
        for element_id, value in additions.items():
            activations[element_id] = max(activations.get(element_id, 0.0), clamp(value))
        ranked = sorted(activations.items(), key=lambda item: (-item[1], item[0]))[: int(cfg["max_activated_elements"])]
        activations = dict(ranked)
    return ActivationResult(activations=activations, paths=paths, hits=tuple(retrieval_hits), steps=int(cfg["activation_steps"]), visited=len(activations))
