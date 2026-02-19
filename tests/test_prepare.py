from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from anomaly_detection.data.load import LoadedFile
from anomaly_detection.data.prepare import prepare


def test_prepare_normalizes_columns_and_fills_missing_label(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  name: skab",
                "  repo_dir: data/raw/skab_repo",
                "  processed_dir: data/processed/skab",
                "  source_url: https://example.invalid/repo.git",
                "loading:",
                "  file_glob: \"**/*.csv\"",
                "  data_subdir: \"\"",
                "  timestamp_column_candidates: [\"timestamp\", \"datetime\"]",
                "  label_column_candidates: [\"anomaly\", \"label\"]",
            ]
        ),
        encoding="utf-8",
    )

    # One file has "label" and one has no label, both use "datetime".
    df_with_label = pd.DataFrame(
        {"datetime": ["2024-01-02", "2024-01-01"], "sensor": [10.0, 11.0], "label": [1, 0]}
    )
    df_without_label = pd.DataFrame(
        {"datetime": ["2024-01-01", "2024-01-02"], "sensor": [1.5, 1.8]}
    )
    loaded = [
        LoadedFile(path=tmp_path / "with_label.csv", df=df_with_label),
        LoadedFile(path=tmp_path / "without_label.csv", df=df_without_label),
    ]

    monkeypatch.setattr("anomaly_detection.data.prepare.find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        "anomaly_detection.data.prepare.acquire_dataset",
        lambda **_: SimpleNamespace(location=SimpleNamespace(raw_data_dir=tmp_path / "raw")),
    )
    monkeypatch.setattr("anomaly_detection.data.prepare.load_dataset", lambda *_: loaded)

    stats = prepare(config_path=config_path)

    assert stats.files == 2
    assert stats.total_rows == 4
    assert stats.anomalies == 1

    processed = tmp_path / "data" / "processed" / "skab"
    out_with = pd.read_csv(processed / "with_label.csv")
    out_without = pd.read_csv(processed / "without_label.csv")

    assert "timestamp" in out_with.columns
    assert "anomaly" in out_with.columns
    assert list(out_with["timestamp"]) == ["2024-01-01", "2024-01-02"]
    assert list(out_with["anomaly"]) == [0, 1]

    assert "timestamp" in out_without.columns
    assert "anomaly" in out_without.columns
    assert list(out_without["anomaly"]) == [0, 0]

    report = tmp_path / "artifacts" / "reports" / "prepare_report.json"
    assert report.exists()
