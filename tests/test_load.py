from pathlib import Path

from anomaly_detection.data.load import load_csv


def test_load_csv_auto_detects_semicolon_delimiter(tmp_path: Path):
    p = tmp_path / "sample.csv"
    p.write_text("datetime;value;anomaly\n2024-01-01 00:00:00;1.0;0\n", encoding="utf-8")

    df = load_csv(p)

    assert list(df.columns) == ["datetime", "value", "anomaly"]
    assert float(df.loc[0, "value"]) == 1.0
