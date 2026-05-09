from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DashboardKpis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_transactions: int = Field(..., ge=0)
    analyzed_transactions: int = Field(..., ge=0)
    known_fraud_rate: float = Field(..., ge=0.0, le=1.0)
    reject_review_rate: float = Field(..., ge=0.0, le=1.0)
    average_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)


class DecisionBreakdownItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: str
    count: int = Field(..., ge=0)


class FraudTrendItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str
    total: int = Field(..., ge=0)
    known_fraud_count: int = Field(..., ge=0)
    analyzed_count: int = Field(..., ge=0)
    rejected_count: int = Field(..., ge=0)


class CategoryRiskItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    total: int = Field(..., ge=0)
    fraud_rate: float = Field(..., ge=0.0, le=1.0)
    average_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)


class HourlyActivityItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour: int = Field(..., ge=0, le=23)
    total: int = Field(..., ge=0)


class TransactionListItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    transaction_datetime: datetime
    merchant: str
    category: str
    amount: float
    risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    decision: str | None = None
    is_fraud: bool | None = None
    model_version: str | None = None


class DashboardOverviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kpis: DashboardKpis
    decision_breakdown: list[DecisionBreakdownItem]
    fraud_trend: list[FraudTrendItem]
    category_risk: list[CategoryRiskItem]
    hourly_activity: list[HourlyActivityItem]
    recent_transactions: list[TransactionListItem]
    alerts: list[str]


class TransactionListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int = Field(..., ge=0)
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)
    transactions: list[TransactionListItem]
