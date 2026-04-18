from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import torch

from training.preprocessing import PreprocessingArtifacts


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_preprocessing_artifacts(output_dir: Path, artifacts: PreprocessingArtifacts) -> None:
    ensure_dir(output_dir)
    joblib.dump(artifacts.scaler, output_dir / "scaler.joblib")
    save_json(output_dir / "category_mappings.json", artifacts.category_mappings)
    save_json(output_dir / "preprocessing_metadata.json", {
        "numeric_columns": artifacts.numeric_columns,
        "categorical_columns": artifacts.categorical_columns,
        "categorical_index_columns": artifacts.categorical_index_columns,
        "numeric_fill_values": artifacts.numeric_fill_values,
        "applied_log1p_columns": artifacts.applied_log1p_columns,
        "pad_index": 0,
        "oov_index": 1,
    })


def save_model_state(output_dir: Path, model: torch.nn.Module) -> None:
    ensure_dir(output_dir)
    torch.save(model.state_dict(), output_dir / "model.pt")

