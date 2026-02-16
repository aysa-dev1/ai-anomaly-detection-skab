from __future__ import annotations

import pandas as pd


def select_feature_columns(df: pd.DataFrame, timestamp_col: str, label_col: str) -> list[str]:
    """exclude timestamp and label, keep the rest as feature candidates """
    drop = {timestamp_col, label_col}
    feature_cols = [c for c in df.columns if c not in drop]
    return feature_cols


def build_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """only numeric feature columns needed"""
    x = df[feature_cols].copy()
    x = x.select_dtypes(include=["number"])
    if x.shape[1] == 0:
        raise ValueError("No numeric feature columns found after selection")
    return x
