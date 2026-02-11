import shutil

import pytest

from anomaly_detection.data.download import ensure_git_available


def test_git_available_or_skip():
    if shutil.which("git") is None:
        pytest.skip("git is not installed in this environment")
    ensure_git_available()
