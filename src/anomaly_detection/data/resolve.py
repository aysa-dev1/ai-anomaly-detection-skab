from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataLocation:
    repo_dir: Path
    raw_data_dir: Path
    csv_count: int


def _count_csvs(path: Path, file_glob: str) -> int:
    return len(list(path.glob(file_glob)))


def resolve_raw_data_dir(repo_dir: Path, file_glob: str, data_subdir: str = "") -> DataLocation:
    repo_dir = repo_dir.resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"Repo directory not found: {repo_dir}")

    data_subdir = data_subdir.strip()
    if data_subdir:
        candidate = (repo_dir / data_subdir).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Configured data_subdir does not exist: {candidate}")
        n = _count_csvs(candidate, file_glob)
        if n == 0:
            raise FileNotFoundError(f"No CSV files found in: {candidate} (glob={file_glob})")
        return DataLocation(repo_dir=repo_dir, raw_data_dir=candidate, csv_count=n)

    def better(candidate: Path, n: int, best_dir: Path | None, best_n: int) -> bool:
        if n > best_n:
            return True
        if n == best_n and best_dir is not None:
            # tie-break: prefer deeper (more specific) directory
            return len(candidate.parts) > len(best_dir.parts)
        return False

    common = ["data", "dataset", "datasets", "Data", "DATA"]
    best_dir: Path | None = None
    best_count = 0

    # 1) check common dirs
    for name in common:
        cand = repo_dir / name
        if cand.exists() and cand.is_dir():
            n = _count_csvs(cand, file_glob)
            if better(cand, n, best_dir, best_count):
                best_dir, best_count = cand, n

    # 2) fallback: scan ONLY subdirectories (not repo_dir itself)
    if best_count == 0:
        for d in (p for p in repo_dir.rglob("*") if p.is_dir()):
            n = _count_csvs(d, file_glob)
            if better(d, n, best_dir, best_count):
                best_dir, best_count = d, n

    if best_dir is None or best_count == 0:
        raise FileNotFoundError(f"Could not locate any CSV files under: {repo_dir} (glob={file_glob})")

    return DataLocation(repo_dir=repo_dir, raw_data_dir=best_dir.resolve(), csv_count=best_count)
