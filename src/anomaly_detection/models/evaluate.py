from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


@dataclass(frozen=True)
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> dict:
        return asdict(self)


def to_binary_anomaly_labels(y_true: np.ndarray) -> np.ndarray:
    # allows bool, int, float
    y = np.asarray(y_true).astype(int)
    return (y != 0).astype(int)


def isolation_forest_pred_to_anomaly(y_pred_iforest: np.ndarray) -> np.ndarray:
    # IsolationForest: -1 anomaly, 1 normal
    y = np.asarray(y_pred_iforest).astype(int)
    return (y == -1).astype(int)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    y_t = to_binary_anomaly_labels(y_true)
    y_p = to_binary_anomaly_labels(y_pred)

    return ClassificationMetrics(
        precision=float(precision_score(y_t, y_p, zero_division=0)),
        recall=float(recall_score(y_t, y_p, zero_division=0)),
        f1=float(f1_score(y_t, y_p, zero_division=0)),
        support=int(y_t.shape[0]),
    )
