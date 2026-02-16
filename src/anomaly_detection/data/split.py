from __future__ import annotations

import pandas as pd


def time_series_split(df: pd.DataFrame, train_ratio: float = 0.7,) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time series DataFrame chronologically into train and test sets

    Assumes the DataFrame is already sorted by time.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    n = len(df)
    if n < 2:
        raise ValueError("DataFrame must contain at least 2 rows")

    split_idx = int(n * train_ratio)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df
