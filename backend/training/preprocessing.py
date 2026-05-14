from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from training.contracts import FeatureSpec
from training.feature_engineering import DERIVED_FEATURES

PAD_INDEX = 0
OOV_INDEX = 1


@dataclass
class PreprocessingArtifacts:
    scaler: RobustScaler
    numeric_columns: list[str]
    categorical_columns: list[str]
    categorical_index_columns: list[str]
    category_mappings: dict[str, dict[str, int]]
    numeric_fill_values: dict[str, float]
    applied_log1p_columns: list[str]


def _append_log_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    added: list[str] = []
    for col in columns:
        if col in out.columns:
            log_col = f"log1p_{col}"
            out[log_col] = np.log1p(np.clip(out[col].astype(float), a_min=0.0, a_max=None))
            added.append(log_col)
    return out, added


def _resolve_numeric_columns(df: pd.DataFrame, spec: FeatureSpec, log_columns: list[str]) -> list[str]:
    derived = [
        DERIVED_FEATURES.hour,
        DERIVED_FEATURES.day_of_week,
        DERIVED_FEATURES.is_weekend,
        DERIVED_FEATURES.month,
        DERIVED_FEATURES.is_night,
        DERIVED_FEATURES.hour_sin,
        DERIVED_FEATURES.hour_cos,
        DERIVED_FEATURES.dow_sin,
        DERIVED_FEATURES.dow_cos,
        DERIVED_FEATURES.geo_distance_km,
        DERIVED_FEATURES.time_since_prev_tx,
        DERIVED_FEATURES.tx_count_1h,
        DERIVED_FEATURES.tx_count_24h,
        DERIVED_FEATURES.amt_mean_24h,
        DERIVED_FEATURES.amt_std_24h,
        DERIVED_FEATURES.amt_zscore_24h,
    ]
    base = [c for c in spec.numeric_columns if c in df.columns]
    present_derived = [c for c in derived if c in df.columns]
    return list(dict.fromkeys(base + present_derived + log_columns))


def _fit_category_mapping(series: pd.Series) -> dict[str, int]:
    unique_values = series.fillna("__MISSING__").astype(str).unique().tolist()
    mapping = {value: idx for idx, value in enumerate(sorted(unique_values), start=2)}
    return mapping


def _encode_categories(
    df: pd.DataFrame,
    categorical_columns: list[str],
    mappings: dict[str, dict[str, int]],
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    index_columns: list[str] = []
    for col in categorical_columns:
        idx_col = f"{col}_idx"
        mapping = mappings[col]
        encoded = (
            out[col]
            .fillna("__MISSING__")
            .astype(str)
            .map(mapping)
            .fillna(OOV_INDEX)
            .astype(np.int64)
        )
        out[idx_col] = encoded
        index_columns.append(idx_col)
    return out, index_columns


def fit_preprocessor(
    train_df: pd.DataFrame,
    *,
    spec: FeatureSpec,
) -> tuple[pd.DataFrame, PreprocessingArtifacts]:
    """Fits normalization and categorical index mappings using train split only."""
    if train_df.empty:
        raise ValueError("Training dataframe is empty.")

    frame, log_columns = _append_log_columns(train_df, spec.log1p_columns)
    numeric_cols = _resolve_numeric_columns(frame, spec, log_columns)
    categorical_cols = [c for c in spec.categorical_columns if c in frame.columns]

    numeric_fill_values = {col: float(frame[col].median()) for col in numeric_cols}
    for col, fill_value in numeric_fill_values.items():
        frame[col] = frame[col].astype(float).fillna(fill_value)

    scaler = RobustScaler()
    frame[numeric_cols] = scaler.fit_transform(frame[numeric_cols]).astype(np.float32)

    category_mappings = {col: _fit_category_mapping(frame[col]) for col in categorical_cols}
    frame, index_cols = _encode_categories(frame, categorical_cols, category_mappings)

    artifacts = PreprocessingArtifacts(
        scaler=scaler,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        categorical_index_columns=index_cols,
        category_mappings=category_mappings,
        numeric_fill_values=numeric_fill_values,
        applied_log1p_columns=log_columns,
    )
    return frame, artifacts


def transform_with_preprocessor(
    df: pd.DataFrame,
    *,
    spec: FeatureSpec,
    artifacts: PreprocessingArtifacts,
) -> pd.DataFrame:
    """Applies fitted train-time normalization and category mappings to new splits."""
    if df.empty:
        return df.copy()

    frame = df.copy()
    for src_col in spec.log1p_columns:
        if src_col in frame.columns:
            frame[f"log1p_{src_col}"] = np.log1p(np.clip(frame[src_col].astype(float), a_min=0.0, a_max=None))

    for col in artifacts.numeric_columns:
        fill_value = artifacts.numeric_fill_values.get(col, 0.0)
        frame[col] = frame[col].astype(float).fillna(fill_value)
    frame[artifacts.numeric_columns] = artifacts.scaler.transform(frame[artifacts.numeric_columns]).astype(np.float32)

    frame, _ = _encode_categories(frame, artifacts.categorical_columns, artifacts.category_mappings)
    return frame
