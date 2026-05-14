from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_col: str = "split"

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")
        for name, value in {
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
        }.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0.0, got {value}")


def temporal_split(
    df: pd.DataFrame,
    *,
    time_col: str,
    config: SplitConfig,
) -> pd.DataFrame:
    """Assigns temporal splits in chronological order."""
    config.validate()

    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")

    out = df.copy()
    out = out.sort_values(time_col).reset_index(drop=True)
    n_rows = len(out)
    if n_rows < 3:
        raise ValueError("Temporal split requires at least 3 rows.")

    train_end = int(n_rows * config.train_ratio)
    val_end = train_end + int(n_rows * config.val_ratio)
    train_end = max(1, min(train_end, n_rows - 2))
    val_end = max(train_end + 1, min(val_end, n_rows - 1))

    split_values = ["train"] * train_end + ["val"] * (val_end - train_end) + ["test"] * (n_rows - val_end)
    out[config.split_col] = split_values
    return out

