from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from anomaly_detection.data.download import DownloadResult, clone_or_pull_repo
from anomaly_detection.data.resolve import DataLocation, resolve_raw_data_dir


@dataclass(frozen=True)
class AcquireResult:
    download: DownloadResult
    location: DataLocation


def acquire_dataset(
    repo_url: str,
    repo_dir: Path,
    file_glob: str = "**/*.csv",
    data_subdir: str = "",
) -> AcquireResult:
    download = clone_or_pull_repo(repo_url=repo_url, target_dir=repo_dir)
    location = resolve_raw_data_dir(repo_dir=download.target_dir, file_glob=file_glob, data_subdir=data_subdir)
    return AcquireResult(download=download, location=location)
