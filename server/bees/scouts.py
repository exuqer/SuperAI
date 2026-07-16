from __future__ import annotations

from server.spaces import MultidimensionalSpace

from .bee_budget import BeeBudget
from .models import BeeTask, NectarComponent, NectarPacket


class ScoutBee:
    bee_type = "scout"

    def run(
        self,
        space: MultidimensionalSpace,
        task: BeeTask,
        budget: BeeBudget,
    ) -> list[NectarPacket]:
        if not budget.try_consume(
            1, bee_type=self.bee_type, operation="locate_entry", task_id=task.task_id
        ):
            return []
        activations = space.activate(
            task.desired_features,
            min_relevance=task.min_relevance,
            limit=min(task.max_candidates, budget.remaining + 1),
        )
        packets: list[NectarPacket] = []
        for activation in activations:
            if not budget.try_consume(
                1, bee_type=self.bee_type, operation="collect", task_id=task.task_id
            ):
                break
            components = tuple(
                NectarComponent(name, activation.dimensions.get(name), score)
                for name, score in activation.activated_dimensions.items()
                if score > 0
            )
            packets.append(
                NectarPacket(
                    origin_space=space.level.value,
                    source_id=activation.object_id,
                    components=components,
                    confidence=activation.confidence,
                    utility=activation.activation,
                    cost=2,
                    provenance=activation.provenance,
                    bee_type=self.bee_type,
                    task_id=task.task_id,
                )
            )
        return packets
