from __future__ import annotations

from datetime import datetime
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


VALID_TRANSACTION_CATEGORIES = {
    "entertainment",
    "food_dining",
    "gas_transport",
    "grocery_net",
    "grocery_pos",
    "health_fitness",
    "home",
    "kids_pets",
    "misc_net",
    "misc_pos",
    "personal_care",
    "shopping_net",
    "shopping_pos",
    "travel",
}


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: float = Field(..., ge=0.0, le=1_000_000.0)
    transaction_datetime: datetime
    merchant: str = Field(..., min_length=1, max_length=128)
    category: str = Field(..., min_length=1, max_length=100)
    gender: str = Field(..., min_length=1, max_length=1)
    state: str = Field(..., min_length=2, max_length=2)
    job: str = Field(..., min_length=1, max_length=128)
    city_population: int = Field(..., ge=0, le=50_000_000)
    customer_latitude: float = Field(..., ge=-90.0, le=90.0)
    customer_longitude: float = Field(..., ge=-180.0, le=180.0)
    merchant_latitude: float = Field(..., ge=-90.0, le=90.0)
    merchant_longitude: float = Field(..., ge=-180.0, le=180.0)
    transactions_last_hour: int = Field(..., ge=0, le=500)
    transactions_last_24h: int = Field(..., ge=0, le=5_000)
    average_amount_24h: float = Field(..., ge=0.0, le=1_000_000.0)
    training_job_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator(
        "merchant",
        "category",
        "gender",
        "state",
        "job",
        "training_job_id",
        mode="before",
    )
    @classmethod
    def strip_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not isinstance(value, str):
            raise ValueError("Value must be text.")
        stripped = value.strip()
        if not stripped:
            raise ValueError("Value must not be empty.")
        return stripped

    @field_validator(
        "amount",
        "customer_latitude",
        "customer_longitude",
        "merchant_latitude",
        "merchant_longitude",
        "average_amount_24h",
    )
    @classmethod
    def reject_non_finite_numbers(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("Value must be a finite number.")
        return value

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: str) -> str:
        if value not in VALID_TRANSACTION_CATEGORIES:
            raise ValueError("Category must match a supported transaction category.")
        return value

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, value: str) -> str:
        if value not in {"F", "M"}:
            raise ValueError("Gender must be F or M.")
        return value

    @field_validator("state")
    @classmethod
    def normalize_state(cls, value: str) -> str:
        normalized = value.upper()
        if len(normalized) != 2 or not normalized.isalpha():
            raise ValueError("State must be a two-letter code.")
        return normalized

    @model_validator(mode="after")
    def validate_transaction_counts(self) -> "PredictRequest":
        if self.transactions_last_hour > self.transactions_last_24h:
            raise ValueError("Transactions last hour cannot exceed transactions last 24h.")
        return self
