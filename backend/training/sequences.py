from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from training.contracts import FeatureSpec


@dataclass(frozen=True)
class SequenceConfig:
    seq_len: int = 30
    stride: int = 1
    split_col: str = "split"


@dataclass(frozen=True)
class SequenceArrays:
    x_num: np.ndarray
    x_cat: np.ndarray
    lengths: np.ndarray
    y: np.ndarray


def build_sequence_arrays(
    df: pd.DataFrame,
    *,
    spec: FeatureSpec,
    numeric_columns: list[str],
    categorical_index_columns: list[str],
    config: SequenceConfig,
    split_value: str,
) -> SequenceArrays:
    """Builds left-padded sequence windows grouped by entity id."""
    if config.seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    if config.stride <= 0:
        raise ValueError("stride must be positive.")

    target_col = spec.target_column
    entity_col = spec.entity_id.columns[0]
    time_col = spec.time_column

    split_df = df[df[config.split_col] == split_value].copy()
    split_df = split_df.sort_values([entity_col, time_col]).reset_index(drop=True)
    if split_df.empty:
        return SequenceArrays(
            x_num=np.zeros((0, config.seq_len, len(numeric_columns)), dtype=np.float32),
            x_cat=np.zeros((0, config.seq_len, len(categorical_index_columns)), dtype=np.int64),
            lengths=np.zeros((0,), dtype=np.int64),
            y=np.zeros((0,), dtype=np.float32),
        )

    x_num_items: list[np.ndarray] = []
    x_cat_items: list[np.ndarray] = []
    lengths: list[int] = []
    y_values: list[float] = []

    for _, group in split_df.groupby(entity_col, sort=False):
        group = group.sort_values(time_col).reset_index(drop=True)
        n_rows = len(group)
        if n_rows == 0:
            continue

        for end_idx in range(0, n_rows, config.stride):
            start_idx = max(0, end_idx - config.seq_len + 1)
            window = group.iloc[start_idx : end_idx + 1]
            length = len(window)
            if length <= 0:
                continue

            num_window = window[numeric_columns].to_numpy(dtype=np.float32)
            cat_window = window[categorical_index_columns].to_numpy(dtype=np.int64)

            num_padded = np.zeros((config.seq_len, len(numeric_columns)), dtype=np.float32)
            cat_padded = np.zeros((config.seq_len, len(categorical_index_columns)), dtype=np.int64)

            num_padded[-length:] = num_window
            cat_padded[-length:] = cat_window

            x_num_items.append(num_padded)
            x_cat_items.append(cat_padded)
            lengths.append(length)
            y_values.append(float(window[target_col].iloc[-1]))

    if not x_num_items:
        return SequenceArrays(
            x_num=np.zeros((0, config.seq_len, len(numeric_columns)), dtype=np.float32),
            x_cat=np.zeros((0, config.seq_len, len(categorical_index_columns)), dtype=np.int64),
            lengths=np.zeros((0,), dtype=np.int64),
            y=np.zeros((0,), dtype=np.float32),
        )

    return SequenceArrays(
        x_num=np.stack(x_num_items, axis=0),
        x_cat=np.stack(x_cat_items, axis=0),
        lengths=np.asarray(lengths, dtype=np.int64),
        y=np.asarray(y_values, dtype=np.float32),
    )

