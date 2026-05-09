from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.core.config import settings
from app.repositories.transaction_repository import PostgresTransactionRepository, TransactionPersistenceError
from app.schemas.dashboard import TransactionListResponse
from app.services.dashboard_service import DashboardService, TransactionListFilters, TransactionSource

router = APIRouter(prefix="/v1/transactions", tags=["transactions"])
dashboard_service = DashboardService(repository=PostgresTransactionRepository(settings.database_url))


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
