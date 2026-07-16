from __future__ import annotations

from typing import Callable

from server.spaces import MultidimensionalSpace

from .bee_budget import BeeBudget
from .models import BeeTask, NectarComponent, NectarPacket


class AssemblyBee:
    bee_type = "assembly"

    def transfer(
        self,
        packet: NectarPacket,
        source_space: MultidimensionalSpace,
        target_space: MultidimensionalSpace,
        task: BeeTask,
        budget: BeeBudget,
        projector: Callable[[NectarPacket], list[str]] | None = None,
    ) -> list[NectarPacket]:
        if not budget.try_consume(
            2, bee_type=self.bee_type, operation="vertical_transfer", task_id=task.task_id
        ):
            return []
        target_ids = (
            projector(packet)
            if projector
            else [
                item["target_id"]
                for item in source_space.down_project(packet.source_id)
                + source_space.up_project(packet.source_id)
            ]
        )
        results: list[NectarPacket] = []
        for target_id in target_ids:
            if target_id not in target_space.objects:
                continue
            cloud = target_space.objects[target_id]
            validation = target_space.validate_candidate(cloud)
            if not validation["valid"]:
                continue
            results.append(
                NectarPacket(
                    origin_space=target_space.level.value,
                    source_id=target_id,
                    components=tuple(
                        NectarComponent(name, value, packet.confidence)
                        for name, value in cloud.dimensions.items()
                    ),
                    confidence=packet.confidence * validation["coverage"],
                    utility=packet.utility * 0.95,
                    cost=2,
                    provenance={**packet.provenance, "assembled_from": packet.source_id},
                    bee_type=self.bee_type,
                    task_id=task.task_id,
                )
            )
        return results
