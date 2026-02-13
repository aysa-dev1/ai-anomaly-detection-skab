from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from anomaly_detection.data.acquire import acquire_dataset
from anomaly_detection.data.load import load_dataset
from anomaly_detection.data.schema import detect_label_column, detect_timestamp_column
from anomaly_detection.utils.config import load_dataset_config
from anomaly_detection.utils.paths import find_repo_root


@dataclass(frozen=True)
class PrepareStats:
    files: int
    total_rows: int
    anomalies: int


def prepare(config_path: Path | None = None) -> PrepareStats:
    root = find_repo_root()
    config_path = config_path or root / "configs" / "dataset.yaml"

    cfg = load_dataset_config(config_path)

    acquired = acquire_dataset(
        repo_url=cfg.source_url,
        repo_dir=cfg.repo_dir,
        file_glob=cfg.file_glob,
        data_subdir=cfg.data_subdir,
    )

    loaded = load_dataset(acquired.location.raw_data_dir, cfg.file_glob)

    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    total_anomalies = 0

    for item in loaded:
        df = item.df.copy()

        ts_col = detect_timestamp_column(df, cfg.timestamp_column_candidates)
        label_col = detect_label_column(df, cfg.label_column_candidates)

        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.sort_values(ts_col).reset_index(drop=True)

        total_rows += len(df)
        total_anomalies += int(df[label_col].sum())

        out_path = cfg.processed_dir / item.path.name
        df.to_csv(out_path, index=False)

    stats = PrepareStats(
        files=len(loaded),
        total_rows=total_rows,
        anomalies=total_anomalies,
    )

    report_path = root / "artifacts" / "reports" / "prepare_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(stats.__dict__, indent=2), encoding="utf-8")

    return stats


if __name__ == "__main__":
    print(prepare())
