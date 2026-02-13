from __future__ import annotations
from typing import Iterable
import pandas as pd

def find_first_matching_column(columns: Iterable[str], candidates: list[str]) -> str:
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in lower_map:
            return lower_map[key]
        
    raise ValueError(f"no matching column found among candidates: {candidates}")


def detect_timestamp_column(df: pd.DataFrame, candidates: list[str]) -> str:
    return find_first_matching_column(df.columns, candidates)

def detect_label_column(df: pd.DataFrame, candidates: list[str]) -> str:
    return find_first_matching_column(df.columns, candidates)