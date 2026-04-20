from __future__ import annotations

import numpy as np

from anomaly_detection.models.evaluate import aggregate_metrics, compute_metrics


def test_compute_metrics_uses_positive_support_and_confusion_counts():
    metrics = compute_metrics(
        y_true=np.array([0, 0, 1, 1]),
        y_pred=np.array([0, 1, 1, 0]),
    )

    assert metrics.support == 2
    assert metrics.tp == 1
    assert metrics.fp == 1
    assert metrics.fn == 1
    assert metrics.tn == 1


def test_aggregate_metrics_uses_global_confusion_counts():
    anomaly_free = compute_metrics(
        y_true=np.array([0, 0, 0, 0]),
        y_pred=np.array([0, 0, 0, 0]),
    )
    anomalous = compute_metrics(
        y_true=np.array([1, 1, 0, 0]),
        y_pred=np.array([1, 1, 1, 0]),
    )

    aggregate = aggregate_metrics([anomaly_free, anomalous])

    assert aggregate.support == 2
    assert aggregate.tp == 2
    assert aggregate.fp == 1
    assert aggregate.fn == 0
    assert aggregate.tn == 5
    assert aggregate.precision == 2 / 3
    assert aggregate.recall == 1.0
    assert aggregate.f1 == 0.8
