import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str
    cors_origins: list[str]
    risk_score_approve_max: float
    risk_score_reject_min: float
    training_default_dataset_path: str
    training_data_root: str
    training_artifacts_root: str
    training_jobs_registry_path: str
    training_python_bin: str


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
    training_default_dataset_path=os.getenv(
        "TRAINING_DEFAULT_DATASET_PATH",
        "/home/soiber/Projetos/tensor-risk-engine/backend/training/data/fraudTrain.csv",
    ),
    training_data_root=os.getenv(
        "TRAINING_DATA_ROOT",
        "/home/soiber/Projetos/tensor-risk-engine/backend/training/data",
    ),
    training_artifacts_root=os.getenv(
        "TRAINING_ARTIFACTS_ROOT",
        "/home/soiber/Projetos/tensor-risk-engine/backend/training/artifacts",
    ),
    training_jobs_registry_path=os.getenv(
        "TRAINING_JOBS_REGISTRY_PATH",
        "/home/soiber/Projetos/tensor-risk-engine/backend/training/artifacts/jobs_registry.json",
    ),
    training_python_bin=os.getenv("TRAINING_PYTHON_BIN", "python"),
)
