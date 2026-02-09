from pathlib import Path

from anomaly_detection.utils.paths import find_repo_root


def test_find_repo_root():
    root = find_repo_root(Path.cwd())
    assert (root / "pyproject.toml").exists()
