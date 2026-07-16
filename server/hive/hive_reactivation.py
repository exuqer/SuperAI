from typing import Iterable

from server.memory import MemoryItem, ThermoGravityMemory


def reactivate(memory: ThermoGravityMemory, topics: Iterable[str]) -> list[MemoryItem]:
    return memory.reactivate(topics)
