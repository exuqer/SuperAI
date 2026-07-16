from server.memory import MemoryItem, ThermoGravityMemory


def eviction_score(memory: ThermoGravityMemory, item: MemoryItem) -> float:
    return memory.eviction_score(item)
