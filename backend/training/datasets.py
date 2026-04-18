"""Dataset utilities for fraud modeling."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

__all__ = [
    "TabularFraudDataset",
    "SequenceFraudDataset",
    "sequence_fraud_collate_fn",
]


def _to_tensor(
    value: Any,
    *,
    name: str,
    dtype: torch.dtype | None = None,
    allow_empty: bool = False,
) -> Tensor:
    """Convert a value to a tensor and validate its basic shape."""

    tensor = torch.as_tensor(value, dtype=dtype)
    if not allow_empty and tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    return tensor


def _validate_lengths(lengths: Tensor, *, sample_count: int) -> Tensor:
    """Validate and normalize sequence lengths."""

    if lengths.ndim != 1:
        raise ValueError("lengths must be a one-dimensional tensor.")
    if lengths.shape[0] != sample_count:
        raise ValueError("lengths must contain one value per sample.")
    if torch.any(lengths <= 0):
        raise ValueError("lengths must contain positive values only.")
    return lengths.to(dtype=torch.long)


class TabularFraudDataset(Dataset[tuple[Tensor, Tensor]]):
    """Dataset for tabular fraud examples.

    Parameters
    ----------
    X:
        Feature matrix with shape ``(n_samples, n_features)``.
    y:
        Target vector with shape ``(n_samples,)`` or ``(n_samples, 1)``.
    """

    def __init__(self, X: Any, y: Any) -> None:
        self._X = _to_tensor(X, name="X", dtype=torch.float32)
        self._y = torch.as_tensor(y)

        if self._X.ndim != 2:
            raise ValueError("X must be a two-dimensional tensor.")
        if self._y.ndim not in (1, 2):
            raise ValueError("y must be a one- or two-dimensional tensor.")
        if self._y.shape[0] != self._X.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        if self._y.ndim == 2:
            if self._y.shape[1] != 1:
                raise ValueError("Two-dimensional y must have shape (n_samples, 1).")
            self._y = self._y.squeeze(-1)

        self._y = self._y.to(dtype=torch.float32)

    def __len__(self) -> int:
        return int(self._X.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self._X[index], self._y[index]


class SequenceFraudDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Dataset for padded sequence fraud examples.

    Parameters
    ----------
    x_num:
        Numeric sequence tensor with shape ``(n_samples, max_seq_len, n_numeric_features)``.
    x_cat:
        Categorical sequence tensor with shape ``(n_samples, max_seq_len, n_categorical_features)``.
    lengths:
        Valid sequence lengths per sample.
    y:
        Binary targets with shape ``(n_samples,)`` or ``(n_samples, 1)``.
    """

    def __init__(self, x_num: Any, x_cat: Any, lengths: Any, y: Any) -> None:
        self._x_num = _to_tensor(x_num, name="x_num", dtype=torch.float32, allow_empty=True)
        self._x_cat = _to_tensor(x_cat, name="x_cat", dtype=torch.long, allow_empty=True)
        self._lengths = _to_tensor(lengths, name="lengths", dtype=torch.long)
        self._y = torch.as_tensor(y)

        if self._x_num.ndim != 3:
            raise ValueError("x_num must be a three-dimensional tensor.")
        if self._x_cat.ndim != 3:
            raise ValueError("x_cat must be a three-dimensional tensor.")
        if self._lengths.shape[0] != self._x_num.shape[0]:
            raise ValueError("lengths must contain one value per sample.")
        if self._x_cat.shape[0] != self._x_num.shape[0]:
            raise ValueError("x_num and x_cat must contain the same number of samples.")
        if self._x_cat.shape[1] != self._x_num.shape[1]:
            raise ValueError("x_num and x_cat must share the same padded sequence length.")
        if self._y.ndim not in (1, 2):
            raise ValueError("y must be a one- or two-dimensional tensor.")
        if self._y.shape[0] != self._x_num.shape[0]:
            raise ValueError("y must contain one value per sample.")
        if self._y.ndim == 2:
            if self._y.shape[1] != 1:
                raise ValueError("Two-dimensional y must have shape (n_samples, 1).")
            self._y = self._y.squeeze(-1)

        self._lengths = _validate_lengths(self._lengths, sample_count=self._x_num.shape[0])
        if torch.any(self._lengths > self._x_num.shape[1]):
            raise ValueError("lengths cannot exceed the padded sequence length.")

        self._y = self._y.to(dtype=torch.float32)

    def __len__(self) -> int:
        return int(self._x_num.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        length = int(self._lengths[index].item())
        return (
            self._x_num[index, :length],
            self._x_cat[index, :length],
            self._lengths[index],
            self._y[index],
        )


def sequence_fraud_collate_fn(
    batch: Sequence[tuple[Tensor, Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Collate variable-length fraud sequences into padded batch tensors."""

    if not batch:
        raise ValueError("batch must not be empty.")

    x_num_list: list[Tensor] = []
    x_cat_list: list[Tensor] = []
    lengths_list: list[Tensor] = []
    y_list: list[Tensor] = []

    for item in batch:
        if len(item) != 4:
            raise ValueError("Each batch item must contain x_num, x_cat, length, and y.")
        x_num, x_cat, length, y = item
        if x_num.ndim != 2:
            raise ValueError("Each x_num item must be two-dimensional.")
        if x_cat.ndim != 2:
            raise ValueError("Each x_cat item must be two-dimensional.")
        if length.ndim != 0:
            raise ValueError("Each length item must be a scalar tensor.")
        if y.ndim not in (0, 1):
            raise ValueError("Each y item must be a scalar tensor or a single-value tensor.")
        x_num_list.append(x_num)
        x_cat_list.append(x_cat)
        lengths_list.append(length.to(dtype=torch.long))
        y_list.append(y.reshape(()).to(dtype=torch.float32))

    lengths = torch.stack(lengths_list, dim=0)
    x_num_padded = pad_sequence(x_num_list, batch_first=True)
    x_cat_padded = pad_sequence(x_cat_list, batch_first=True)
    y_batch = torch.stack(y_list, dim=0)

    if torch.any(lengths <= 0):
        raise ValueError("All sequence lengths must be positive.")
    if torch.any(lengths > x_num_padded.shape[1]):
        raise ValueError("Sequence lengths cannot exceed the padded batch length.")

    return x_num_padded, x_cat_padded, lengths, y_batch
