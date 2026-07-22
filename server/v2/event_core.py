"""Public V3.0 event API."""

from .event_graph import EventGraphPipeline
from .graph_learning import signature_similarity
from .graph_repository import stable_id


UniversalEventPipeline = EventGraphPipeline


def graph_compatible(left, right) -> float:
    """Return compatibility of two sparse observation signatures."""
    return signature_similarity(left, right)


__all__ = [
    "EventGraphPipeline",
    "UniversalEventPipeline",
    "graph_compatible",
    "stable_id",
]
