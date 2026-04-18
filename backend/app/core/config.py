import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str
    cors_origins: list[str]
    risk_score_approve_max: float
    risk_score_reject_min: float


def _parse_cors_origins(raw_value: str) -> list[str]:
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    return values or ["*"]


def _parse_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        return default
    return max(0.0, min(1.0, value))


settings = Settings(
    app_name=os.getenv("APP_NAME", "BTC Tensor Lab API"),
    cors_origins=_parse_cors_origins(os.getenv("CORS_ORIGINS", "*")),
    risk_score_approve_max=_parse_float_env("RISK_SCORE_APPROVE_MAX", 0.3),
    risk_score_reject_min=_parse_float_env("RISK_SCORE_REJECT_MIN", 0.7),
)
