from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DerivedFeatureNames:
    """Canonical derived feature names."""

    hour: str = "hour"
    day_of_week: str = "day_of_week"
    is_weekend: str = "is_weekend"
    month: str = "month"
    is_night: str = "is_night"
    hour_sin: str = "hour_sin"
    hour_cos: str = "hour_cos"
    dow_sin: str = "dow_sin"
    dow_cos: str = "dow_cos"
    geo_distance_km: str = "geo_distance_km"
    time_since_prev_tx: str = "time_since_prev_tx"
    tx_count_1h: str = "tx_count_1h"
    tx_count_24h: str = "tx_count_24h"
    amt_mean_24h: str = "amt_mean_24h"
    amt_std_24h: str = "amt_std_24h"
    amt_zscore_24h: str = "amt_zscore_24h"


DERIVED_FEATURES = DerivedFeatureNames()
EPSILON = 1e-8


def haversine_distance_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Vectorized Haversine distance in kilometers."""
    earth_radius_km = 6371.0088

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return earth_radius_km * c


def _append_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    dt = out[time_col].dt

    out[DERIVED_FEATURES.hour] = dt.hour.astype(np.float32)
    out[DERIVED_FEATURES.day_of_week] = dt.weekday.astype(np.float32)
    out[DERIVED_FEATURES.is_weekend] = dt.weekday.isin([5, 6]).astype(np.float32)
    out[DERIVED_FEATURES.month] = dt.month.astype(np.float32)
    out[DERIVED_FEATURES.is_night] = dt.hour.isin([0, 1, 2, 3, 4, 5]).astype(np.float32)

    hour_angle = 2.0 * np.pi * (out[DERIVED_FEATURES.hour] / 24.0)
    dow_angle = 2.0 * np.pi * (out[DERIVED_FEATURES.day_of_week] / 7.0)

    out[DERIVED_FEATURES.hour_sin] = np.sin(hour_angle).astype(np.float32)
    out[DERIVED_FEATURES.hour_cos] = np.cos(hour_angle).astype(np.float32)
    out[DERIVED_FEATURES.dow_sin] = np.sin(dow_angle).astype(np.float32)
    out[DERIVED_FEATURES.dow_cos] = np.cos(dow_angle).astype(np.float32)
    return out


def _append_distance_feature(
    df: pd.DataFrame,
    customer_lat_col: str = "lat",
    customer_lon_col: str = "long",
    merchant_lat_col: str = "merch_lat",
    merchant_lon_col: str = "merch_long",
) -> pd.DataFrame:
    out = df.copy()
    out[DERIVED_FEATURES.geo_distance_km] = haversine_distance_km(
        out[customer_lat_col].to_numpy(dtype=np.float64),
        out[customer_lon_col].to_numpy(dtype=np.float64),
        out[merchant_lat_col].to_numpy(dtype=np.float64),
        out[merchant_lon_col].to_numpy(dtype=np.float64),
    ).astype(np.float32)
    return out


def _causal_behavior_features_for_group(
    group_df: pd.DataFrame,
    time_col: str,
    amount_col: str,
) -> pd.DataFrame:
    group = group_df.sort_values(time_col).copy()
    times = group[time_col].to_numpy()
    amounts = group[amount_col].to_numpy(dtype=np.float64)

    n_rows = len(group)
    time_since_prev_tx = np.zeros(n_rows, dtype=np.float32)
    tx_count_1h = np.zeros(n_rows, dtype=np.float32)
    tx_count_24h = np.zeros(n_rows, dtype=np.float32)
    amt_mean_24h = np.zeros(n_rows, dtype=np.float32)
    amt_std_24h = np.zeros(n_rows, dtype=np.float32)
    amt_zscore_24h = np.zeros(n_rows, dtype=np.float32)

    one_hour = np.timedelta64(1, "h")
    twenty_four_hours = np.timedelta64(24, "h")

    idx_1h: deque[int] = deque()
    idx_24h: deque[int] = deque()

    sum_24h = 0.0
    sq_sum_24h = 0.0

    for i in range(n_rows):
        current_time = times[i]
        current_amount = amounts[i]

        if i > 0:
            delta_seconds = (current_time - times[i - 1]) / np.timedelta64(1, "s")
            time_since_prev_tx[i] = float(max(delta_seconds, 0.0))

        while idx_1h and (current_time - times[idx_1h[0]]) > one_hour:
            idx_1h.popleft()

        while idx_24h and (current_time - times[idx_24h[0]]) > twenty_four_hours:
            pop_idx = idx_24h.popleft()
            pop_amount = amounts[pop_idx]
            sum_24h -= pop_amount
            sq_sum_24h -= pop_amount * pop_amount

        tx_count_1h[i] = float(len(idx_1h))
        tx_count_24h[i] = float(len(idx_24h))

        if idx_24h:
            count = float(len(idx_24h))
            mean = sum_24h / count
            variance = max((sq_sum_24h / count) - (mean * mean), 0.0)
            std = sqrt(variance)

            amt_mean_24h[i] = float(mean)
            amt_std_24h[i] = float(std)
            if std > EPSILON:
                amt_zscore_24h[i] = float((current_amount - mean) / std)

        idx_1h.append(i)
        idx_24h.append(i)
        sum_24h += current_amount
        sq_sum_24h += current_amount * current_amount

    group[DERIVED_FEATURES.time_since_prev_tx] = time_since_prev_tx
    group[DERIVED_FEATURES.tx_count_1h] = tx_count_1h
    group[DERIVED_FEATURES.tx_count_24h] = tx_count_24h
    group[DERIVED_FEATURES.amt_mean_24h] = amt_mean_24h
    group[DERIVED_FEATURES.amt_std_24h] = amt_std_24h
    group[DERIVED_FEATURES.amt_zscore_24h] = amt_zscore_24h
    return group


def append_causal_behavior_features(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    amount_col: str = "amt",
) -> pd.DataFrame:
    """Adds strictly causal behavior features grouped by entity."""
    out = (
        df.groupby(entity_col, group_keys=False, sort=False)
        .apply(lambda g: _causal_behavior_features_for_group(g, time_col, amount_col))
        .reset_index(drop=True)
    )
    return out


def build_feature_frame(
    raw_df: pd.DataFrame,
    *,
    entity_col: str,
    time_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Builds the full feature frame with causal and temporal derived features."""
    required = {entity_col, time_col, target_col, "amt", "lat", "long", "merch_lat", "merch_long"}
    missing = required.difference(raw_df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns for feature engineering: {missing_str}")

    frame = raw_df.copy()
    frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
    frame = frame.dropna(subset=[time_col]).sort_values([entity_col, time_col]).reset_index(drop=True)

    frame = _append_time_features(frame, time_col=time_col)
    frame = _append_distance_feature(frame)
    frame = append_causal_behavior_features(frame, entity_col=entity_col, time_col=time_col, amount_col="amt")

    frame[DERIVED_FEATURES.geo_distance_km] = frame[DERIVED_FEATURES.geo_distance_km].fillna(0.0)
    frame[DERIVED_FEATURES.time_since_prev_tx] = frame[DERIVED_FEATURES.time_since_prev_tx].fillna(0.0)
    frame[DERIVED_FEATURES.amt_zscore_24h] = frame[DERIVED_FEATURES.amt_zscore_24h].replace([np.inf, -np.inf], 0.0)
    frame[DERIVED_FEATURES.amt_zscore_24h] = frame[DERIVED_FEATURES.amt_zscore_24h].fillna(0.0)
    return frame
