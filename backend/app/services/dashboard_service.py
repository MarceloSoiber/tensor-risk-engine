from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from typing import Protocol


class TransactionSource(StrEnum):
    ALL = "all"
    ANALYZED = "analyzed"
    HISTORICAL = "historical"


@dataclass(frozen=True)
class TransactionListFilters:
    limit: int = 50
    offset: int = 0
    source: TransactionSource = TransactionSource.ALL
    decision: str | None = None
    category: str | None = None


@dataclass(frozen=True)
class DashboardOverview:
    kpis: dict[str, object]
    decision_breakdown: list[dict[str, object]]
    fraud_trend: list[dict[str, object]]
    category_risk: list[dict[str, object]]
    hourly_activity: list[dict[str, object]]
    recent_transactions: list[dict[str, object]]
    alerts: list[str]


class DashboardRepository(Protocol):
    def fetch_dashboard_overview_rows(self, *, start_at: datetime, limit: int) -> dict[str, object]:
        ...

    def list_transactions(self, filters: TransactionListFilters) -> tuple[int, list[dict[str, object]]]:
        ...


class DashboardService:
    def __init__(self, repository: DashboardRepository) -> None:
        self._repository = repository

    def get_overview(self, *, days: int = 30, limit: int = 50) -> DashboardOverview:
        normalized_days = max(1, min(days, 365))
        normalized_limit = max(1, min(limit, 200))
        start_at = datetime.combine(
            date.today() - timedelta(days=normalized_days - 1),
            time.min,
        )
        rows = self._repository.fetch_dashboard_overview_rows(start_at=start_at, limit=normalized_limit)

        kpis = self._build_kpis(rows)
        decision_breakdown = self._normalize_decision_breakdown(rows.get("decision_breakdown", []))
        fraud_trend = self._normalize_fraud_trend(rows.get("fraud_trend", []))
        category_risk = self._normalize_category_risk(rows.get("category_risk", []))
        hourly_activity = self._normalize_hourly_activity(rows.get("hourly_activity", []))
        recent_transactions = list(rows.get("recent_transactions", []))
        alerts = self._build_alerts(kpis, decision_breakdown, category_risk)

        return DashboardOverview(
            kpis=kpis,
            decision_breakdown=decision_breakdown,
            fraud_trend=fraud_trend,
            category_risk=category_risk,
            hourly_activity=hourly_activity,
            recent_transactions=recent_transactions,
            alerts=alerts,
        )

    def list_transactions(self, filters: TransactionListFilters) -> tuple[int, list[dict[str, object]]]:
        return self._repository.list_transactions(filters)

    @staticmethod
    def _build_kpis(rows: dict[str, object]) -> dict[str, object]:
        totals = dict(rows.get("kpis", {}) or {})
        total_transactions = int(totals.get("total_transactions") or 0)
        analyzed_transactions = int(totals.get("analyzed_transactions") or 0)
        known_labeled = int(totals.get("known_labeled_transactions") or 0)
        known_fraud = int(totals.get("known_fraud_transactions") or 0)
        reject_review = int(totals.get("reject_review_transactions") or 0)
        scored = int(totals.get("scored_transactions") or 0)
        risk_sum = float(totals.get("risk_score_sum") or 0.0)

        return {
            "total_transactions": total_transactions,
            "analyzed_transactions": analyzed_transactions,
            "known_fraud_rate": _safe_ratio(known_fraud, known_labeled),
            "reject_review_rate": _safe_ratio(reject_review, analyzed_transactions),
            "average_risk_score": risk_sum / scored if scored else None,
        }

    @staticmethod
    def _normalize_decision_breakdown(rows: object) -> list[dict[str, object]]:
        counts = {"approve": 0, "review": 0, "reject": 0}
        for row in rows if isinstance(rows, list) else []:
            decision = str(row.get("decision") or "")
            if decision in counts:
                counts[decision] = int(row.get("count") or 0)
        return [{"decision": decision, "count": count} for decision, count in counts.items()]

    @staticmethod
    def _normalize_fraud_trend(rows: object) -> list[dict[str, object]]:
        return [
            {
                "date": str(row.get("date")),
                "total": int(row.get("total") or 0),
                "known_fraud_count": int(row.get("known_fraud_count") or 0),
                "analyzed_count": int(row.get("analyzed_count") or 0),
                "rejected_count": int(row.get("rejected_count") or 0),
            }
            for row in rows
            if isinstance(row, dict)
        ]

    @staticmethod
    def _normalize_category_risk(rows: object) -> list[dict[str, object]]:
        normalized: list[dict[str, object]] = []
        for row in rows if isinstance(rows, list) else []:
            total = int(row.get("total") or 0)
            fraud_count = int(row.get("fraud_count") or 0)
            scored = int(row.get("scored") or 0)
            risk_sum = float(row.get("risk_score_sum") or 0.0)
            normalized.append(
                {
                    "category": str(row.get("category") or "unknown"),
                    "total": total,
                    "fraud_rate": _safe_ratio(fraud_count, total),
                    "average_risk_score": risk_sum / scored if scored else None,
                }
            )
        return normalized

    @staticmethod
    def _normalize_hourly_activity(rows: object) -> list[dict[str, object]]:
        counts = {hour: 0 for hour in range(24)}
        for row in rows if isinstance(rows, list) else []:
            hour = int(row.get("hour") or 0)
            if 0 <= hour <= 23:
                counts[hour] = int(row.get("total") or 0)
        return [{"hour": hour, "total": count} for hour, count in counts.items()]

    @staticmethod
    def _build_alerts(
        kpis: dict[str, object],
        decision_breakdown: list[dict[str, object]],
        category_risk: list[dict[str, object]],
    ) -> list[str]:
        alerts: list[str] = []
        reject_count = next((int(item["count"]) for item in decision_breakdown if item["decision"] == "reject"), 0)
        review_count = next((int(item["count"]) for item in decision_breakdown if item["decision"] == "review"), 0)

        if reject_count:
            alerts.append(f"{reject_count} transactions were rejected in the selected window.")
        if review_count:
            alerts.append(f"{review_count} transactions require manual review.")
        if float(kpis["known_fraud_rate"]) >= 0.05:
            alerts.append("Known fraud rate is above the 5% monitoring threshold.")
        if category_risk:
            riskiest = max(category_risk, key=lambda item: float(item["fraud_rate"]))
            if float(riskiest["fraud_rate"]) > 0:
                alerts.append(f"{riskiest['category']} has the highest known fraud concentration.")
        if not alerts:
            alerts.append("No elevated fraud activity detected in the selected window.")
        return alerts


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator
