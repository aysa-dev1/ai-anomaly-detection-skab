from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class LoadedFile:
    path: Path
    df: pd.DataFrame


def iter_csv_files(raw_dir: Path, file_glob: str) -> Iterable[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    yield from sorted(raw_dir.glob(file_glob))


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_dataset(raw_dir: Path, file_glob: str = "**/*.csv") -> list[LoadedFile]:
    files = list(iter_csv_files(raw_dir, file_glob))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {raw_dir} (glob={file_glob})")

    loaded: list[LoadedFile] = []
    for p in files:
        df = load_csv(p)
        loaded.append(LoadedFile(path=p, df=df))
    return loaded
