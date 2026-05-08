from __future__ import annotations

import pandas as pd


def select_feature_columns(df: pd.DataFrame,
                           timestamp_col: str, 
                           label_col: str,
                           metadata_cols: list[str]) -> list[str]:
    """exclude timestamp and label, keep the rest as feature candidates """
    drop = {timestamp_col, label_col, *(metadata_cols or [])}
    feature_cols = [c for c in df.columns if c not in drop]
    return feature_cols

def build_engineered_features(df: pd.DataFrame, feature_cols: list[str], rolling_window: int):
    df_engineered = df.copy()

    for col in feature_cols:
        df_engineered[f"{col}_roll_mean"] = df_engineered[col].rolling(window=rolling_window, min_periods=1).mean()

        df_engineered[f"{col}_roll_std"] = df_engineered[col].rolling(window=rolling_window, min_periods=1).std().fillna(0)

        df_engineered[f"{col}_diff"] = df_engineered[col].diff().fillna(0)

    return df_engineered

def build_feature_matrix(df: pd.DataFrame, feature_cols: list[str], expected_features: list[str] | None = None,) -> pd.DataFrame:
    """only numeric feature columns needed"""
    x = df[feature_cols].copy()
    x = x.select_dtypes(include=["number"])
    if expected_features is not None:
        x = x.reindex(columns=expected_features, fill_value=0)
    if x.shape[1] == 0:
        raise ValueError("No numeric feature columns found after selection")
    return x
