from pathlib import Path

import pytest


from anomaly_detection.data.resolve import resolve_raw_data_dir

def test_resolve_prefers_configured_subdir(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "mydata").mkdir(parents=True)
    (repo / "mydata" / "a.csv").write_text("x\n1\n")

    loc = resolve_raw_data_dir(repo_dir=repo, file_glob="**/*.csv")
    assert loc.raw_data_dir.name == "mydata"
    assert loc.csv_count == 1


def test_resolve_raises_if_no_csvs(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_raw_data_dir(repo_dir=repo, file_glob="**/*.csv")
    
    