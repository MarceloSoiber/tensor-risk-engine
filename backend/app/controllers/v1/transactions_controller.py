from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status

from app.core.config import settings
from app.features.feature_builder import FeatureBuilder
from app.ml.inference.risk_engine import RiskInferenceEngine
from app.ml.loaders.model_loader import ModelLoader
from app.repositories.transaction_repository import PostgresTransactionRepository, TransactionPersistenceError
from app.schemas.dashboard import (
    FraudTestImportRequest,
    TransactionImportJobResponse,
    TransactionListResponse,
)
from app.services.dashboard_service import DashboardService, TransactionListFilters, TransactionSource
from app.services.risk_service import RiskService
from app.services.transaction_import_service import ImportResult, TransactionImportError, TransactionImportService

router = APIRouter(prefix="/v1/transactions", tags=["transactions"])
transaction_repository = PostgresTransactionRepository(settings.database_url)
dashboard_service = DashboardService(repository=transaction_repository)
model_loader = ModelLoader()
risk_service = RiskService(
    repository=transaction_repository,
    feature_builder=FeatureBuilder(),
    inference_engine=RiskInferenceEngine(model_loader=model_loader),
    model_loader=model_loader,
)
transaction_import_service = TransactionImportService(settings.database_url)

ImportJobStatus = Literal["queued", "running", "succeeded", "failed"]


@dataclass
class TransactionImportJobRecord:
    job_id: str
    status: ImportJobStatus
    dataset_path: str
    created_at: datetime
    updated_at: datetime
    processed_rows: int = 0
    imported_rows: int = 0
    analyzed_rows: int = 0
    error: str | None = None
    finished_at: datetime | None = None

    def to_response(self) -> TransactionImportJobResponse:
        return TransactionImportJobResponse(
            job_id=self.job_id,
            status=self.status,
            dataset_path=self.dataset_path,
            processed_rows=self.processed_rows,
            imported_rows=self.imported_rows,
            analyzed_rows=self.analyzed_rows,
            error=self.error,
            created_at=self.created_at,
            updated_at=self.updated_at,
            finished_at=self.finished_at,
        )


_import_jobs: dict[str, TransactionImportJobRecord] = {}
_import_jobs_lock = threading.RLock()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@router.get("", response_model=TransactionListResponse)
def list_transactions(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    source: TransactionSource = Query(default=TransactionSource.ALL),
    decision: str | None = Query(default=None, pattern="^(approve|review|reject)$"),
    category: str | None = Query(default=None, min_length=1, max_length=100),
) -> TransactionListResponse:
    filters = TransactionListFilters(
        limit=limit,
        offset=offset,
        source=source,
        decision=decision,
        category=category,
    )

    try:
        total, transactions = dashboard_service.list_transactions(filters)
    except TransactionPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transactions are unavailable.",
        ) from exc

    return TransactionListResponse(
        total=total,
        limit=limit,
        offset=offset,
        transactions=transactions,
    )


@router.post(
    "/import/fraud-test",
    response_model=TransactionImportJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_fraud_test_import(
    payload: FraudTestImportRequest,
    background_tasks: BackgroundTasks,
) -> TransactionImportJobResponse:
    dataset_path = _resolve_fraud_test_dataset_path()

    with _import_jobs_lock:
        running_job = next((job for job in _import_jobs.values() if job.status in {"queued", "running"}), None)
        if running_job is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Transaction import job '{running_job.job_id}' is already running.",
            )

        job = TransactionImportJobRecord(
            job_id=uuid.uuid4().hex,
            status="queued",
            dataset_path=str(dataset_path),
            created_at=_utc_now(),
            updated_at=_utc_now(),
        )
        _import_jobs[job.job_id] = job

    background_tasks.add_task(
        _run_fraud_test_import,
        job.job_id,
        dataset_path,
        payload.batch_size,
        payload.training_job_id,
    )
    return job.to_response()


@router.get("/import-jobs/{job_id}", response_model=TransactionImportJobResponse)
def get_transaction_import_job(job_id: str) -> TransactionImportJobResponse:
    with _import_jobs_lock:
        job = _import_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Import job '{job_id}' was not found.")
        return job.to_response()


def _run_fraud_test_import(
    job_id: str,
    dataset_path: Path,
    batch_size: int,
    training_job_id: str | None,
) -> None:
    _update_import_job(job_id, status="running")
    try:
        result = transaction_import_service.import_and_analyze_csv(
            dataset_path=dataset_path,
            batch_size=batch_size,
            risk_service=risk_service,
            training_job_id=training_job_id,
            progress_callback=lambda progress: _update_import_job(job_id, result=progress),
        )
    except (FileNotFoundError, TransactionImportError, ValueError) as exc:
        _update_import_job(job_id, status="failed", error=str(exc), finished=True)
        return

    _update_import_job(job_id, status="succeeded", result=result, finished=True)


def _update_import_job(
    job_id: str,
    *,
    status: ImportJobStatus | None = None,
    result: ImportResult | None = None,
    error: str | None = None,
    finished: bool = False,
) -> None:
    with _import_jobs_lock:
        job = _import_jobs.get(job_id)
        if job is None:
            return

        if status is not None:
            job.status = status
        if result is not None:
            job.processed_rows = result.processed_rows
            job.imported_rows = result.imported_rows
            job.analyzed_rows = result.analyzed_rows
        if error is not None:
            job.error = error
        if finished:
            job.finished_at = _utc_now()
        job.updated_at = _utc_now()


def _resolve_fraud_test_dataset_path() -> Path:
    data_root = Path(settings.training_data_root).resolve()
    dataset_path = (data_root / "fraudTest.csv").resolve()
    try:
        dataset_path.relative_to(data_root)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="fraudTest.csv must be inside the training data root.") from exc

    if not dataset_path.exists() or not dataset_path.is_file():
        raise HTTPException(status_code=404, detail="fraudTest.csv was not found in the training data directory.")
    return dataset_path
