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


TrainingPipeline = TrainingPipelineV2
