"""Single source of truth for thermogravitational working-memory capacity."""

from __future__ import annotations

from typing import Any, Dict, Iterable


WORKING_CLASSES = {
    "context": "context_cells",
    "role_candidate": "reasoning_cells",
    "semantic_bridge": "reasoning_cells",
    "reasoning_support": "reasoning_cells",
    "memory_source": "memory_sources",
}


def capacity_pressure(active_total: int, capacity: int) -> float:
    occupancy = active_total / max(int(capacity), 1)
    if occupancy <= .60:
        return 0.0
    return min(1.0, ((occupancy - .60) / .40) ** 2)


def get_working_occupancy(cells: Iterable[Dict[str, Any]], capacity: int) -> Dict[str, Any]:
    counts = {"query_cells": 0, "reasoning_cells": 0, "memory_sources": 0, "context_cells": 0}
    for cell in cells:
        bucket = WORKING_CLASSES.get(str(cell.get("component_class") or ""))
        if bucket:
            counts[bucket] += 1
    active_total = sum(counts.values())
    max_cells = max(int(capacity), 1)
    occupancy = active_total / max_cells
    return {
        "capacity": max_cells,
        **counts,
        "active_total": active_total,
        "occupancy": round(occupancy, 6),
        "pressure": round(capacity_pressure(active_total, max_cells), 6),
        "eviction_threshold": .75,
        "compression_threshold": .60,
    }
