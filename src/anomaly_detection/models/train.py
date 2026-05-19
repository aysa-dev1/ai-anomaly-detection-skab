from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from anomaly_detection.data.split import time_series_split
from anomaly_detection.features.build_features import build_feature_matrix, select_feature_columns, build_engineered_features
from anomaly_detection.models.baseline import IsolationForestConfig, build_isolation_forest
from anomaly_detection.models.evaluate import (
    aggregate_metrics,
    compute_metrics,
    isolation_forest_pred_to_anomaly,
)
from anomaly_detection.utils.config import load_train_config
from anomaly_detection.utils.logging import get_logger
from anomaly_detection.utils.paths import find_repo_root
from anomaly_detection.utils.seeds import set_global_seed


def load_prepared_csvs(processed_dir: Path) -> list[Path]:
    return sorted(processed_dir.glob("*.csv"))


def train_baseline_on_dataset(
    processed_dir: Path,
    artifacts_models_dir: Path,
    artifacts_metrics_dir: Path,
    timestamp_col: str,
    label_col: str,
    train_ratio: float = 0.7,
    cfg: IsolationForestConfig | None = None,
    metadata_cols: list[str] | None = None,
    rolling_window: int = 20,
) -> dict:
    cfg = cfg or IsolationForestConfig()

    artifacts_models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_metrics_dir.mkdir(parents=True, exist_ok=True)

    files = load_prepared_csvs(processed_dir)
    if not files:
        raise FileNotFoundError(f"No prepared CSVs found in {processed_dir}")

    per_file: dict[str, dict] = {}
    file_metrics = []
    model = None

    for p in files:
        df = pd.read_csv(p)

        train_df, test_df = time_series_split(df, train_ratio=train_ratio)

        feature_cols = select_feature_columns(df, timestamp_col=timestamp_col, label_col=label_col, metadata_cols=metadata_cols)

        train_engineered = build_engineered_features(train_df, feature_cols, rolling_window)
        test_engineered = build_engineered_features(test_df, feature_cols, rolling_window)

        all_feature_cols = [c for c in train_engineered.columns
                            if c not in {timestamp_col, label_col, *(metadata_cols or [])}]

        x_train = build_feature_matrix(train_engineered, all_feature_cols)
        x_test = build_feature_matrix(test_engineered, all_feature_cols)

        contamination = cfg.contamination
        if contamination == "from_data":
            rate = float(np.asarray(train_df[label_col]).mean())
            contamination = float(np.clip(rate, 1e-4, 0.5))

        model = build_isolation_forest(
            IsolationForestConfig(
                n_estimators=cfg.n_estimators,
                contamination=contamination,
                random_state=cfg.random_state,
            )
        )
        model.fit(x_train)

        raw_pred = model.predict(x_test)
        y_pred = isolation_forest_pred_to_anomaly(raw_pred)
        y_true = test_df[label_col].to_numpy()

        m = compute_metrics(y_true=y_true, y_pred=y_pred)
        file_metrics.append(m)

        per_file[p.name] = m.to_dict()

    agg = aggregate_metrics(file_metrics).to_dict()

    metrics = {
        "model": "IsolationForest",
        "train_ratio": train_ratio,
        "config": {
            "n_estimators": cfg.n_estimators,
            "contamination": cfg.contamination,
            "random_state": cfg.random_state,
        },
        "aggregate": agg,
        "per_file": per_file,
    }

    metrics_path = artifacts_metrics_dir / "baseline_isolation_forest.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    model_path = artifacts_models_dir / "isolation_forest.joblib"
    joblib.dump(model, model_path)

    return metrics


def _resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def train_from_config(config_path: str | Path) -> dict[str, Any]:
    root = find_repo_root()
    cfg = load_train_config(config_path)

    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    logging_cfg = cfg.get("logging", {})
    features_cfg = cfg.get("features", {})

    logger = get_logger(__name__, level=logging_cfg.get("level", "INFO"))

    seed = int(training_cfg.get("seed", 42))
    set_global_seed(seed)
    logger.info("Global seed set to %s", seed)

    processed_dir = _resolve_path(root, training_cfg["processed_dir"])
    artifacts_models_dir = _resolve_path(root, training_cfg["artifacts_models_dir"])
    artifacts_metrics_dir = _resolve_path(root, training_cfg["artifacts_metrics_dir"])

    model_name = model_cfg.get("name", "isolation_forest")
    if model_name != "isolation_forest":
        raise ValueError(f"Unsupported model name: {model_name}")

    model = IsolationForestConfig(
        n_estimators=int(model_cfg.get("n_estimators", 200)),
        contamination=model_cfg.get("contamination", "auto"),
        random_state=int(model_cfg.get("random_state", seed)),
    )

    metadata_cols = training_cfg.get("metadata_cols", [])
    rolling_window = features_cfg.get("rolling_window", 20)

    metrics = train_baseline_on_dataset(
        processed_dir=processed_dir,
        artifacts_models_dir=artifacts_models_dir,
        artifacts_metrics_dir=artifacts_metrics_dir,
        timestamp_col=training_cfg["timestamp_col"],
        label_col=training_cfg["label_col"],
        train_ratio=float(training_cfg.get("train_ratio", 0.7)),
        cfg=model,
        metadata_cols=metadata_cols,
        rolling_window=rolling_window,
    )

    logger.info("Training completed. Aggregate metrics: %s", metrics["aggregate"])
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train anomaly detection baseline model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config YAML.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_from_config(args.config)
