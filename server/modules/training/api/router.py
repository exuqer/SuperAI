"""Training HTTP routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from server.modules.training.api.dto import (
    BatchPreviewRequest,
    BatchReferenceRequest,
    CommitRequest,
    ReprocessRequest,
    RetractionRequest,
    StageRequest,
    TrainRequest,
)
from server.modules.training.application.services import TrainingService


router = APIRouter(prefix="/training", tags=["training"])


def get_training_service() -> TrainingService:
    return TrainingService()


@router.post("/learn")
async def train(
    request: TrainRequest,
    service: TrainingService = Depends(get_training_service),
) -> dict[str, Any]:
    return service.pipeline.train(
        request.text,
        source_type=request.source_type,
        independent_key=request.independent_key,
        domain_key=request.domain_key,
    )


@router.post("/stage")
async def stage(
    request: StageRequest,
    service: TrainingService = Depends(get_training_service),
) -> dict[str, Any]:
    return service.stage(
        request.text,
        source_type=request.source_type,
        independent_key=request.independent_key,
        domain_key=request.domain_key,
    )


@router.post("/commit")
async def commit(
    request: CommitRequest,
    service: TrainingService = Depends(get_training_service),
) -> dict[str, Any]:
    try:
        return service.commit(
            request.source_id,
            manual_validation=request.manual_validation,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail="staging item not found") from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post("/retract")
async def retract(
    request: RetractionRequest,
    service: TrainingService = Depends(get_training_service),
) -> dict[str, Any]:
    try:
        return service.retract(request.source_id, request.reason)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="staging item not found") from error


@router.post("/reprocess")
async def reprocess(
    request: ReprocessRequest,
    service: TrainingService = Depends(get_training_service),
) -> dict[str, Any]:
    try:
        return service.reprocess(request.source_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="staging item not found") from error


@router.post("/batches/preview")
async def preview_batch(
    request: BatchPreviewRequest,
    service: TrainingService = Depends(get_training_service),
) -> dict[str, Any]:
    try:
        return service.preview_batch(request.sources, request.config)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/batches/commit")
async def commit_batch(
    request: BatchReferenceRequest,
    service: TrainingService = Depends(get_training_service),
) -> dict[str, Any]:
    try:
        return service.commit_batch(request.batch_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="batch not found") from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post("/batches/rollback")
async def rollback_batch(
    request: BatchReferenceRequest,
    service: TrainingService = Depends(get_training_service),
) -> dict[str, Any]:
    try:
        return service.rollback_batch(request.batch_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="batch not found") from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
