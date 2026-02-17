from __future__ import annotations

from pathlib import Path

import pandas as pd

from anomaly_detection.models.train import train_baseline_on_dataset


def test_train_baseline_smoke(tmp_path: Path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    # minimal prepared-like csv
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3, 4, 5, 6],
            "sensor_1": [0.1, 0.2, 0.15, 5.0, 0.2, 0.1],
            "anomaly": [0, 0, 0, 1, 0, 0],
        }
    )
    (processed_dir / "scenario_1.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    artifacts_models = tmp_path / "artifacts_models"
    artifacts_metrics = tmp_path / "artifacts_metrics"

    metrics = train_baseline_on_dataset(
        processed_dir=processed_dir,
        artifacts_models_dir=artifacts_models,
        artifacts_metrics_dir=artifacts_metrics,
        timestamp_col="timestamp",
        label_col="anomaly",
        train_ratio=0.7,
    )

    assert "aggregate" in metrics
    assert (artifacts_metrics / "baseline_isolation_forest.json").exists()
    assert (artifacts_models / "isolation_forest.joblib").exists()
