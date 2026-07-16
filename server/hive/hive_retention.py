from server.memory import MemoryItem


def retention_score(item: MemoryItem) -> float:
    return min(
        1.0,
        item.retention * 0.35
        + item.mass * 0.1
        + item.user_priority * 0.2
        + item.unresolved_support * 0.2
        + item.recency * 0.15,
    )
