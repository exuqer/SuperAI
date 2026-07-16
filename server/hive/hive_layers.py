from __future__ import annotations

from typing import Any

from server.memory import MemoryLayer, ThermoGravityMemory


class HiveLayers:
    def __init__(self, memory: ThermoGravityMemory) -> None:
        self.memory = memory

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {
            layer.value: [
                item.to_dict() for item in self.memory.items.values() if item.layer == layer
            ]
            for layer in MemoryLayer
        }
