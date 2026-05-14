from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float


def find_threshold_for_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_precision: float,
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    for idx, precision_value in enumerate(precision[:-1]):
        if precision_value >= min_precision:
            return float(thresholds[idx])
    return 0.5


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    y_true = y_true.astype(int)
    y_pred = (y_score >= threshold).astype(int)

    if len(np.unique(y_true)) < 2:
        roc_auc = 0.0
        pr_auc = 0.0
    else:
        roc_auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    return ClassificationMetrics(
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=float(threshold),
    )

