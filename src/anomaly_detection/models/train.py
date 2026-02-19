from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from anomaly_detection.data.split import time_series_split
from anomaly_detection.features.build_features import build_feature_matrix, select_feature_columns
from anomaly_detection.models.baseline import IsolationForestConfig, build_isolation_forest
from anomaly_detection.models.evaluate import compute_metrics, isolation_forest_pred_to_anomaly


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
) -> dict:
    cfg = cfg or IsolationForestConfig()

    artifacts_models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_metrics_dir.mkdir(parents=True, exist_ok=True)

    files = load_prepared_csvs(processed_dir)
    if not files:
        raise FileNotFoundError(f"No prepared CSVs found in {processed_dir}")

    per_file: dict[str, dict] = {}
    all_prec, all_rec, all_f1, total_support = 0.0, 0.0, 0.0, 0

    # train on each file, evaluate, and store the "last" Model
    # tbd: clean global training
    model = None

    for p in files:
        df = pd.read_csv(p)

        train_df, test_df = time_series_split(df, train_ratio=train_ratio)

        feature_cols = select_feature_columns(df, timestamp_col=timestamp_col, label_col=label_col)
        x_train = build_feature_matrix(train_df, feature_cols)
        x_test = build_feature_matrix(test_df, feature_cols)

        model = build_isolation_forest(cfg)
        model.fit(x_train)

        raw_pred = model.predict(x_test)
        y_pred = isolation_forest_pred_to_anomaly(raw_pred)
        y_true = test_df[label_col].to_numpy()

        m = compute_metrics(y_true=y_true, y_pred=y_pred)

        per_file[p.name] = m.to_dict()

        all_prec += m.precision * m.support
        all_rec += m.recall * m.support
        all_f1 += m.f1 * m.support
        total_support += m.support

    # support-weighted aggregate
    agg = {
        "precision": (all_prec / total_support) if total_support else 0.0,
        "recall": (all_rec / total_support) if total_support else 0.0,
        "f1": (all_f1 / total_support) if total_support else 0.0,
        "support": total_support,
    }

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

    # Save last trained model (minimal approach)
    if model is not None:
        model_path = artifacts_models_dir / "isolation_forest.joblib"
        joblib.dump(model, model_path)

    return metrics

if __name__ == "__main__":
    metrics = train_baseline_on_dataset(
        processed_dir=Path("data/processed/skab"),
        artifacts_models_dir=Path("artifacts/models"),
        artifacts_metrics_dir=Path("artifacts/metrics"),
        timestamp_col="timestamp",
        label_col="anomaly",
        train_ratio=0.7,
    )
    print("Aggregate metrics:", metrics["aggregate"])

