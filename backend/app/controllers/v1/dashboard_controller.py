from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.core.config import settings
from app.repositories.transaction_repository import PostgresTransactionRepository, TransactionPersistenceError
from app.schemas.dashboard import DashboardOverviewResponse
from app.services.dashboard_service import DashboardService

router = APIRouter(prefix="/v1/dashboard", tags=["dashboard"])
dashboard_service = DashboardService(repository=PostgresTransactionRepository(settings.database_url))


@router.get("/overview", response_model=DashboardOverviewResponse)
def get_dashboard_overview(
    days: int = Query(default=3650, ge=1, le=3650),
    limit: int = Query(default=50, ge=1, le=200),
) -> DashboardOverviewResponse:
    try:
        overview = dashboard_service.get_overview(days=days, limit=limit)
    except TransactionPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dashboard data is unavailable.",
        ) from exc

    return DashboardOverviewResponse.model_validate(
        {
            "kpis": overview.kpis,
            "decision_breakdown": overview.decision_breakdown,
            "fraud_trend": overview.fraud_trend,
            "category_risk": overview.category_risk,
            "hourly_activity": overview.hourly_activity,
            "recent_transactions": overview.recent_transactions,
            "alerts": overview.alerts,
        }
    )
