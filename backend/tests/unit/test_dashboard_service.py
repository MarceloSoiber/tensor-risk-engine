from __future__ import annotations

from datetime import datetime

from app.services.dashboard_service import DashboardService, TransactionListFilters


class FakeDashboardRepository:
    def fetch_dashboard_overview_rows(self, *, start_at: datetime, limit: int) -> dict[str, object]:  # noqa: ARG002
        return {
            "kpis": {
                "total_transactions": 4,
                "analyzed_transactions": 2,
                "known_labeled_transactions": 3,
                "known_fraud_transactions": 1,
                "reject_review_transactions": 1,
                "scored_transactions": 2,
                "risk_score_sum": 1.0,
            },
            "decision_breakdown": [{"decision": "reject", "count": 1}, {"decision": "approve", "count": 1}],
            "fraud_trend": [
                {
                    "date": "2020-06-21",
                    "total": 4,
                    "known_fraud_count": 1,
                    "analyzed_count": 2,
                    "rejected_count": 1,
                }
            ],
            "category_risk": [{"category": "grocery_pos", "total": 4, "fraud_count": 1, "scored": 2, "risk_score_sum": 1.0}],
            "hourly_activity": [{"hour": 3, "total": 4}],
            "recent_transactions": [],
        }

    def list_transactions(self, filters: TransactionListFilters) -> tuple[int, list[dict[str, object]]]:  # noqa: ARG002
        return 0, []


def test_dashboard_service_builds_overview_metrics() -> None:
    overview = DashboardService(FakeDashboardRepository()).get_overview(days=30, limit=50)

    assert overview.kpis["total_transactions"] == 4
    assert overview.kpis["known_fraud_rate"] == 1 / 3
    assert overview.kpis["reject_review_rate"] == 0.5
    assert overview.kpis["average_risk_score"] == 0.5
    assert overview.decision_breakdown == [
        {"decision": "approve", "count": 1},
        {"decision": "review", "count": 0},
        {"decision": "reject", "count": 1},
    ]
    assert len(overview.hourly_activity) == 24
    assert overview.hourly_activity[3]["total"] == 4
    assert overview.alerts
