from __future__ import annotations

from server.spaces import MultidimensionalSpace

from .bee_budget import BeeBudget
from .models import BeeTask, NectarComponent, NectarPacket


class WorkerBee:
    bee_type = "worker"

    def run(
        self,
        space: MultidimensionalSpace,
        task: BeeTask,
        sources: list[NectarPacket],
        budget: BeeBudget,
    ) -> list[NectarPacket]:
        packets: list[NectarPacket] = []
        visited = {packet.source_id for packet in sources}
        for source in sorted(sources, key=lambda packet: (-packet.utility, packet.source_id)):
            if budget.remaining < 2 or source.source_id not in space.objects:
                break
            budget.consume(1, bee_type=self.bee_type, operation="expand", task_id=task.task_id)
            for candidate in space.expand(
                source.source_id, limit=min(task.max_candidates, budget.remaining)
            ):
                candidate_id = candidate["object_id"]
                if candidate_id in visited:
                    continue
                visited.add(candidate_id)
                if not budget.try_consume(
                    1, bee_type=self.bee_type, operation="evaluate", task_id=task.task_id
                ):
                    break
                activation = space.activate(
                    task.desired_features,
                    min_relevance=0.0,
                    limit=len(space.objects),
                )
                match = next((item for item in activation if item.object_id == candidate_id), None)
                if not match or match.activation < task.min_relevance * 0.8:
                    continue
                packets.append(
                    NectarPacket(
                        origin_space=space.level.value,
                        source_id=candidate_id,
                        components=tuple(
                            NectarComponent(name, match.dimensions.get(name), score)
                            for name, score in match.activated_dimensions.items()
                            if score > 0
                        ),
                        confidence=match.confidence,
                        utility=match.activation,
                        cost=2,
                        provenance=match.provenance,
                        bee_type=self.bee_type,
                        task_id=task.task_id,
                    )
                )
        return sorted(packets, key=lambda packet: (-packet.utility, packet.source_id))[
            : task.max_candidates
        ]
