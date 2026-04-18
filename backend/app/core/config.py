import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str
    cors_origins: list[str]


def _parse_cors_origins(raw_value: str) -> list[str]:
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    return values or ["*"]


settings = Settings(
    app_name=os.getenv("APP_NAME", "BTC Tensor Lab API"),
    cors_origins=_parse_cors_origins(os.getenv("CORS_ORIGINS", "*")),
)
