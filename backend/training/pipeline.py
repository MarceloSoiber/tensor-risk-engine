from __future__ import annotations

from pathlib import Path

import pandas as pd

from training.contracts import FeatureSpec, load_feature_spec
from training.feature_engineering import build_feature_frame
from training.preprocessing import PreprocessingArtifacts, fit_preprocessor, transform_with_preprocessor
from training.split import SplitConfig, temporal_split


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def run_data_pipeline(
    *,
    dataset_path: str | Path,
    spec_path: str | Path | None = None,
) -> tuple[pd.DataFrame, FeatureSpec, PreprocessingArtifacts]:
    spec = load_feature_spec(spec_path)
    raw_df = load_dataset(dataset_path)

    frame = build_feature_frame(
        raw_df,
        entity_col=spec.entity_id.columns[0],
        time_col=spec.time_column,
        target_col=spec.target_column,
    )

    drop_candidates = [c for c in spec.drop_columns if c in frame.columns]
    if drop_candidates:
        frame = frame.drop(columns=drop_candidates)

    frame = temporal_split(
        frame,
        time_col=spec.time_column,
        config=SplitConfig(
            train_ratio=spec.split.train_ratio,
            val_ratio=spec.split.validation_ratio,
            test_ratio=spec.split.test_ratio,
            split_col="split",
        ),
    )

    train_mask = frame["split"] == "train"
    train_df = frame[train_mask].copy()
    transformed_train, artifacts = fit_preprocessor(train_df, spec=spec)

    remaining = frame[~train_mask].copy()
    transformed_remaining = transform_with_preprocessor(remaining, spec=spec, artifacts=artifacts)
    transformed_frame = pd.concat([transformed_train, transformed_remaining], axis=0).sort_index()
    return transformed_frame.reset_index(drop=True), spec, artifacts

