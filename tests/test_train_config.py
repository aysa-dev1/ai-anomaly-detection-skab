from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from anomaly_detection.models.train import train_from_config


def _write_prepared_csv(processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3, 4, 5, 6],
            "sensor_1": [0.1, 0.2, 0.15, 5.0, 0.2, 0.1],
            "anomaly": [0, 0, 0, 1, 0, 0],
        }
    )
    (processed_dir / "scenario_1.csv").write_text(df.to_csv(index=False), encoding="utf-8")


def test_train_from_config_writes_outputs(tmp_path: Path, monkeypatch):
    _write_prepared_csv(tmp_path / "data" / "processed" / "skab")

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "training:",
                "  processed_dir: data/processed/skab",
                "  artifacts_models_dir: artifacts/models",
                "  artifacts_metrics_dir: artifacts/metrics",
                "  timestamp_col: timestamp",
                "  label_col: anomaly",
                "  train_ratio: 0.7",
                "  seed: 42",
                "model:",
                "  name: isolation_forest",
                "  n_estimators: 50",
                "  contamination: auto",
                "  random_state: 42",
                "logging:",
                "  level: INFO",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("anomaly_detection.models.train.find_repo_root", lambda: tmp_path)

    metrics = train_from_config(cfg_path)

    assert "aggregate" in metrics
    assert (tmp_path / "artifacts" / "metrics" / "baseline_isolation_forest.json").exists()
    assert (tmp_path / "artifacts" / "models" / "isolation_forest.joblib").exists()


def test_train_from_config_rejects_unsupported_model(tmp_path: Path, monkeypatch):
    _write_prepared_csv(tmp_path / "data" / "processed" / "skab")

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "training:",
                "  processed_dir: data/processed/skab",
                "  artifacts_models_dir: artifacts/models",
                "  artifacts_metrics_dir: artifacts/metrics",
                "  timestamp_col: timestamp",
                "  label_col: anomaly",
                "  train_ratio: 0.7",
                "model:",
                "  name: one_class_svm",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("anomaly_detection.models.train.find_repo_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="Unsupported model name"):
        train_from_config(cfg_path)
