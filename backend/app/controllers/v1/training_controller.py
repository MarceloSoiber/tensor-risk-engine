from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.schemas.training import (
    TrainingJobListResponse,
    TrainingJobResponse,
    TrainingJobStartRequest,
)
from app.services.training_job_service import (
    TrainingJobConflictError,
    TrainingJobNotFoundError,
    TrainingJobService,
    TrainingJobValidationError,
)

router = APIRouter(prefix="/v1/training", tags=["training"])
training_job_service = TrainingJobService()


@router.post("/jobs", response_model=TrainingJobResponse, status_code=status.HTTP_202_ACCEPTED)
def start_training_job(payload: TrainingJobStartRequest) -> TrainingJobResponse:
    try:
        job = training_job_service.start_job(payload.model_dump())
    except TrainingJobConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except TrainingJobValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return TrainingJobResponse.model_validate(job)


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
def get_training_job(job_id: str) -> TrainingJobResponse:
    try:
        job = training_job_service.get_job(job_id)
    except TrainingJobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return TrainingJobResponse.model_validate(job)


@router.get("/jobs", response_model=TrainingJobListResponse)
def list_training_jobs() -> TrainingJobListResponse:
    jobs = training_job_service.list_jobs()
    return TrainingJobListResponse(jobs=[TrainingJobResponse.model_validate(item) for item in jobs])


@router.post("/jobs/{job_id}/cancel", response_model=TrainingJobResponse)
def cancel_training_job(job_id: str) -> TrainingJobResponse:
    try:
        job = training_job_service.cancel_job(job_id)
    except TrainingJobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return TrainingJobResponse.model_validate(job)

