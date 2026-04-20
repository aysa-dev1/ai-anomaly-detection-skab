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
    tp: int
    fp: int
    fn: int
    tn: int

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


def compute_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    y_t = to_binary_anomaly_labels(y_true)
    y_p = to_binary_anomaly_labels(y_pred)
    tp = int(((y_t == 1) & (y_p == 1)).sum())
    fp = int(((y_t == 0) & (y_p == 1)).sum())
    fn = int(((y_t == 1) & (y_p == 0)).sum())
    tn = int(((y_t == 0) & (y_p == 0)).sum())
    return tp, fp, fn, tn


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    y_t = to_binary_anomaly_labels(y_true)
    y_p = to_binary_anomaly_labels(y_pred)
    tp, fp, fn, tn = compute_confusion_counts(y_t, y_p)

    return ClassificationMetrics(
        precision=float(precision_score(y_t, y_p, zero_division=0)),
        recall=float(recall_score(y_t, y_p, zero_division=0)),
        f1=float(f1_score(y_t, y_p, zero_division=0)),
        support=int(y_t.sum()),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def aggregate_metrics(items: list[ClassificationMetrics]) -> ClassificationMetrics:
    tp = sum(item.tp for item in items)
    fp = sum(item.fp for item in items)
    fn = sum(item.fn for item in items)
    tn = sum(item.tn for item in items)

    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0

    return ClassificationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        support=tp + fn,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )
