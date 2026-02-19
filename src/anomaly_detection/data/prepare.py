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
    if "dataset" not in cfg or "loading" not in cfg:
        raise ValueError(f"Config missing required sections 'dataset' and 'loading': {config_path}")

    dataset_cfg = cfg["dataset"]
    loading_cfg = cfg["loading"]

    repo_dir = root / Path(dataset_cfg["repo_dir"])
    processed_dir = root / Path(dataset_cfg["processed_dir"])
    file_glob = loading_cfg.get("file_glob", "**/*.csv")
    data_subdir = loading_cfg.get("data_subdir", "")
    timestamp_candidates = loading_cfg.get(
        "timestamp_column_candidates", ["timestamp", "time", "datetime", "date"]
    )
    label_candidates = loading_cfg.get(
        "label_column_candidates", ["anomaly", "label", "target", "is_anomaly"]
    )
    canonical_timestamp = timestamp_candidates[0]
    canonical_label = label_candidates[0]

    acquired = acquire_dataset(
        repo_url=dataset_cfg["source_url"],
        repo_dir=repo_dir,
        file_glob=file_glob,
        data_subdir=data_subdir,
    )

    loaded = load_dataset(acquired.location.raw_data_dir, file_glob)

    processed_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    total_anomalies = 0

    for item in loaded:
        df = item.df.copy()

        ts_col = detect_timestamp_column(df, timestamp_candidates)
        try:
            label_col = detect_label_column(df, label_candidates)
        except ValueError:
            # SKAB anomaly-free split has no explicit label; treat as all normal.
            df[canonical_label] = 0
            label_col = canonical_label

        df[ts_col] = pd.to_datetime(df[ts_col])
        if ts_col != canonical_timestamp:
            df = df.rename(columns={ts_col: canonical_timestamp})
            ts_col = canonical_timestamp
        if label_col != canonical_label:
            df = df.rename(columns={label_col: canonical_label})
            label_col = canonical_label
        df = df.sort_values(ts_col).reset_index(drop=True)

        total_rows += len(df)
        total_anomalies += int(df[label_col].sum())

        out_path = processed_dir / item.path.name
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
