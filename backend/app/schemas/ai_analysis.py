from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


TransactionSourceFilter = Literal["all", "analyzed", "historical"]
DecisionFilter = Literal["approve", "review", "reject"]


class AiAnalysisFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: TransactionSourceFilter = "all"
    decision: DecisionFilter | None = None
    category: str | None = Field(default=None, min_length=1, max_length=100)
    limit: int = Field(default=25, ge=1, le=200)

    @field_validator("category", mode="before")
    @classmethod
    def strip_category(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not isinstance(value, str):
            raise ValueError("Category must be text.")
        stripped = value.strip()
        if not stripped:
            return None
        return stripped


class AiAnalysisQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=3, max_length=1000)
    filters: AiAnalysisFilters = Field(default_factory=AiAnalysisFilters)

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Question must be text.")
        stripped = value.strip()
        if not stripped:
            raise ValueError("Question must not be empty.")
        return stripped


class AiAnalysisTransaction(BaseModel):
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


class AiAnalysisDataSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_count: int = Field(..., ge=0)
    filtered_total: int = Field(..., ge=0)
    source: TransactionSourceFilter
    decision: DecisionFilter | None = None
    category: str | None = None
    average_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    total_amount: float = Field(..., ge=0.0)


class AiAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_id: int
    question: str
    answer: str
    insights: list[str]
    recommended_actions: list[str]
    data_summary: AiAnalysisDataSummary
    transactions: list[AiAnalysisTransaction]
    model: str
    created_at: datetime


class AiAnalysisHistoryItem(AiAnalysisResponse):
    status: str
    error_message: str | None = None


class AiAnalysisHistoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int = Field(..., ge=0)
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)
    analyses: list[AiAnalysisHistoryItem]


class AiAnalysisObservabilityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tracing_enabled: bool
    project: str
    dashboard_url: str
    traceable_provider: bool
    provider_note: str
