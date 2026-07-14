"""Training HTTP routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from server.modules.training.api.dto import TrainRequest
from server.modules.training.application.services import TrainingService


router = APIRouter(prefix="/api/v2/training", tags=["training"])


def get_training_service() -> TrainingService:
    return TrainingService()


@router.post("/learn")
async def train(
    request: TrainRequest,
    service: TrainingService = Depends(get_training_service),
) -> dict[str, Any]:
    return service.train(request.text)
