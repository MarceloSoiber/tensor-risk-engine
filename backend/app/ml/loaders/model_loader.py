from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from app.core.config import BACKEND_ROOT, settings


@dataclass(frozen=True)
class BaselineModelArtifacts:
    model: Any
    scaler: Any
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    categorical_index_columns: list[str]
    category_mappings: dict[str, dict[str, int]]
    numeric_fill_values: dict[str, float]
    decision_threshold: float
    model_version: str


class ModelLoader:
    def __init__(self, fallback_model_version: str = "heuristic-v1") -> None:
        self._fallback_model_version = fallback_model_version
        self._baseline_artifacts: BaselineModelArtifacts | None = None
        self._baseline_artifacts_dir: Path | None = None

    def load_current_baseline_model(self) -> BaselineModelArtifacts | None:
        artifacts_dir = self._find_latest_succeeded_baseline_artifacts_dir()
        if self._baseline_artifacts is not None and self._baseline_artifacts_dir == artifacts_dir:
            return self._baseline_artifacts

        if artifacts_dir is None:
            self._baseline_artifacts = None
            self._baseline_artifacts_dir = None
            return None

        try:
            artifacts = self._load_baseline_artifacts(artifacts_dir)
        except (OSError, ValueError, KeyError, TypeError):
            self._baseline_artifacts = None
            self._baseline_artifacts_dir = None
            return None

        self._baseline_artifacts = artifacts
        self._baseline_artifacts_dir = artifacts_dir
        return artifacts

    def load_baseline_model_by_job_id(self, job_id: str) -> BaselineModelArtifacts | None:
        normalized_job_id = str(job_id).strip()
        if not normalized_job_id:
            return None

        artifacts_dir = self._find_succeeded_baseline_artifacts_dir_by_job_id(normalized_job_id)
        if artifacts_dir is None:
            return None

        try:
            return self._load_baseline_artifacts(artifacts_dir)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def load_model_version(self) -> str:
        artifacts = self.load_current_baseline_model()
        if artifacts is None:
            return self._fallback_model_version
        return artifacts.model_version

    def load_decision_threshold(self) -> float | None:
        artifacts = self.load_current_baseline_model()
        if artifacts is None:
            return None
        return artifacts.decision_threshold

    def _find_latest_succeeded_baseline_artifacts_dir(self) -> Path | None:
        registry_path = Path(settings.training_jobs_registry_path)
        if not registry_path.exists():
            return None

        registry = _read_json(registry_path)
        jobs = registry.get("jobs", [])
        if not isinstance(jobs, list):
            return None

        baseline_jobs = [
            job
            for job in jobs
            if isinstance(job, dict)
            and job.get("status") == "succeeded"
            and job.get("model_type") == "baseline"
            and job.get("artifacts_dir")
        ]
        if not baseline_jobs:
            return None

        latest_job = max(baseline_jobs, key=lambda job: str(job.get("finished_at") or job.get("updated_at") or ""))
        artifacts_dir = _resolve_artifacts_path(str(latest_job["artifacts_dir"]))
        if not (artifacts_dir / "baseline_model.joblib").exists():
            return None
        return artifacts_dir

    def _find_succeeded_baseline_artifacts_dir_by_job_id(self, job_id: str) -> Path | None:
        registry_path = Path(settings.training_jobs_registry_path)
        if not registry_path.exists():
            return None

        registry = _read_json(registry_path)
        jobs = registry.get("jobs", [])
        if not isinstance(jobs, list):
            return None

        target_job = next(
            (
                job
                for job in jobs
                if isinstance(job, dict)
                and str(job.get("job_id")) == job_id
                and job.get("status") == "succeeded"
                and job.get("model_type") == "baseline"
                and job.get("artifacts_dir")
            ),
            None,
        )
        if target_job is None:
            return None

        artifacts_dir = _resolve_artifacts_path(str(target_job["artifacts_dir"]))
        if not (artifacts_dir / "baseline_model.joblib").exists():
            return None
        return artifacts_dir

    def _load_baseline_artifacts(self, artifacts_dir: Path) -> BaselineModelArtifacts:
        metadata = _read_json(artifacts_dir / "training_metadata.json")
        preprocessing = _read_json(artifacts_dir / "preprocessing_metadata.json")
        thresholds = _read_json(artifacts_dir / "thresholds.json")

        feature_columns = _require_string_list(metadata, "feature_columns")
        numeric_columns = _require_string_list(preprocessing, "numeric_columns")
        categorical_columns = _require_string_list(preprocessing, "categorical_columns")
        categorical_index_columns = _require_string_list(preprocessing, "categorical_index_columns")
        category_mappings = _require_category_mappings(_read_json(artifacts_dir / "category_mappings.json"))
        numeric_fill_values = _require_float_mapping(preprocessing, "numeric_fill_values")
        decision_threshold = float(thresholds.get("decision_threshold", 0.5))

        return BaselineModelArtifacts(
            model=joblib.load(artifacts_dir / "baseline_model.joblib"),
            scaler=joblib.load(artifacts_dir / "scaler.joblib"),
            feature_columns=feature_columns,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            categorical_index_columns=categorical_index_columns,
            category_mappings=category_mappings,
            numeric_fill_values=numeric_fill_values,
            decision_threshold=max(0.0, min(1.0, decision_threshold)),
            model_version=f"baseline:{artifacts_dir.name}",
        )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _resolve_artifacts_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    if path.is_absolute() and path.parts[:2] == ("/", "app"):
        return BACKEND_ROOT / Path(*path.parts[2:])
    return path


def _require_string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload[key]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings.")
    return value


def _require_float_mapping(payload: dict[str, Any], key: str) -> dict[str, float]:
    value = payload[key]
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object.")
    return {str(item_key): float(item_value) for item_key, item_value in value.items()}


def _require_category_mappings(payload: dict[str, Any]) -> dict[str, dict[str, int]]:
    mappings: dict[str, dict[str, int]] = {}
    for column, mapping in payload.items():
        if not isinstance(mapping, dict):
            raise ValueError("category_mappings values must be objects.")
        mappings[str(column)] = {str(raw_value): int(index) for raw_value, index in mapping.items()}
    return mappings
