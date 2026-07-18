"""Training application service."""

from __future__ import annotations

from typing import Any

from server.modules.model.infrastructure.repository import ModelRepository
from server.v2.training import TrainingPipelineV2


class TrainingService:
    def __init__(self, repository: ModelRepository | None = None) -> None:
        self.pipeline = TrainingPipelineV2(repository or ModelRepository())

    def train(self, text: str) -> dict[str, Any]:
        return self.pipeline.train(text)

    def stage(self, text: str, **options: Any) -> dict[str, Any]:
        return self.pipeline.stage(text, **options)

    def commit(
        self,
        staging_id: str,
        *,
        manual_validation: bool = False,
    ) -> dict[str, Any]:
        return self.pipeline.commit(
            staging_id,
            manual_validation=manual_validation,
        )

    def retract(self, staging_id: str, reason: str = "") -> dict[str, Any]:
        return self.pipeline.retract(staging_id, reason=reason)

    def reprocess(self, staging_id: str) -> dict[str, Any]:
        return self.pipeline.reprocess(staging_id)

    def preview_batch(
        self,
        sources: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.pipeline.preview_batch(sources, config=config)

    def commit_batch(self, batch_id: str) -> dict[str, Any]:
        return self.pipeline.commit_batch(batch_id)

    def rollback_batch(self, batch_id: str) -> dict[str, Any]:
        return self.pipeline.rollback_batch(batch_id)


TrainingPipeline = TrainingPipelineV2
