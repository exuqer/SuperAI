from .base import MultidimensionalSpace, SpaceLevel


class EventSpace(MultidimensionalSpace):
    level = SpaceLevel.EVENT
    dimensions = (
        "agent",
        "action",
        "object",
        "location",
        "time",
        "polarity",
        "modality",
        "causality",
        "dialogue_relevance",
        "topic_relevance",
    )
    weights = {
        "agent": 1.15,
        "action": 1.3,
        "object": 1.2,
        "location": 0.9,
        "time": 0.7,
        "polarity": 1.2,
        "modality": 0.8,
        "causality": 0.8,
        "dialogue_relevance": 0.65,
        "topic_relevance": 0.75,
    }
