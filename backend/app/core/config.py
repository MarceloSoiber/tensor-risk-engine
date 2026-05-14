import os
from dataclasses import dataclass
from pathlib import Path


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
    database_url: str
    local_llm_base_url: str
    local_llm_model: str
    local_llm_api_key: str
    openrouter_base_url: str
    openrouter_model: str | None
    openrouter_api_key: str | None
    ai_analysis_max_transactions: int
    langsmith_tracing: bool
    langsmith_api_key: str | None
    langsmith_project: str
    langsmith_endpoint: str


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


def _parse_int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_DATA_ROOT = (BACKEND_ROOT / "training" / "data").resolve()
DEFAULT_TRAINING_ARTIFACTS_ROOT = (BACKEND_ROOT / "training" / "artifacts").resolve()


settings = Settings(
    app_name=os.getenv("APP_NAME", "Credit Card Fraud Detection API"),
    cors_origins=_parse_cors_origins(os.getenv("CORS_ORIGINS", "*")),
    risk_score_approve_max=_parse_float_env("RISK_SCORE_APPROVE_MAX", 0.3),
    risk_score_reject_min=_parse_float_env("RISK_SCORE_REJECT_MIN", 0.7),
    training_default_dataset_path=os.getenv(
        "TRAINING_DEFAULT_DATASET_PATH",
        str(DEFAULT_TRAINING_DATA_ROOT / "fraudTrain.csv"),
    ),
    training_data_root=os.getenv(
        "TRAINING_DATA_ROOT",
        str(DEFAULT_TRAINING_DATA_ROOT),
    ),
    training_artifacts_root=os.getenv(
        "TRAINING_ARTIFACTS_ROOT",
        str(DEFAULT_TRAINING_ARTIFACTS_ROOT),
    ),
    training_jobs_registry_path=os.getenv(
        "TRAINING_JOBS_REGISTRY_PATH",
        str(DEFAULT_TRAINING_ARTIFACTS_ROOT / "jobs_registry.json"),
    ),
    training_python_bin=os.getenv("TRAINING_PYTHON_BIN", "python"),
    database_url=os.getenv(
        "DATABASE_URL",
        "postgresql://fraud_user:fraud_password@localhost:5432/fraud_detection",
    ),
    local_llm_base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/api/v1/chat"),
    local_llm_model=os.getenv("LOCAL_LLM_MODEL", "openai/gpt-oss-20b"),
    local_llm_api_key=os.getenv("LOCAL_LLM_API_KEY", "lm-studio"),
    openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions"),
    openrouter_model=_optional_env("OPENROUTER_MODEL"),
    openrouter_api_key=_optional_env("OPENROUTER_API_KEY"),
    ai_analysis_max_transactions=_parse_int_env("AI_ANALYSIS_MAX_TRANSACTIONS", 25, minimum=1, maximum=200),
    langsmith_tracing=_parse_bool_env("LANGSMITH_TRACING") or _parse_bool_env("LANGCHAIN_TRACING_V2"),
    langsmith_api_key=_optional_env("LANGSMITH_API_KEY") or _optional_env("LANGCHAIN_API_KEY"),
    langsmith_project=os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "credit-card-fraud-detection",
    langsmith_endpoint=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
)
