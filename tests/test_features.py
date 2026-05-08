import pandas as pd
import pytest

from anomaly_detection.features.build_features import (
    build_feature_matrix,
    select_feature_columns,
    build_engineered_features,
)


def test_select_feature_columns_excludes_timestamp_and_label():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2020-01-01", periods=3),
            "sensor_1": [1.0, 2.0, 3.0],
            "label": [0, 0, 1],
        }
    )
    cols = select_feature_columns(df, timestamp_col="ts", label_col="label", metadata_cols=[])
    assert "ts" not in cols
    assert "label" not in cols
    assert "sensor_1" in cols

def test_select_feature_columns_excludes_metadata_cols():
    df = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", periods=3),
        "sensor_1": [1.0, 2.0, 3.0],
        "label": [0, 0, 1],
        "changepoint": [0, 0, 1],
    })

    cols = select_feature_columns(df, timestamp_col="ts", label_col="label", metadata_cols=["changepoint"])

    assert "changepoint" not in cols
    assert "sensor_1" in cols


def test_build_feature_matrix_keeps_only_numeric():
    df = pd.DataFrame(
        {
            "sensor_1": [1.0, 2.0, 3.0],
            "status": ["ok", "ok", "fail"],
        }
    )
    x = build_feature_matrix(df, feature_cols=["sensor_1", "status"])
    assert list(x.columns) == ["sensor_1"]
    assert x.shape == (3, 1)


def test_build_feature_matrix_raises_if_no_numeric():
    df = pd.DataFrame({"status": ["ok", "ok"]})
    with pytest.raises(ValueError, match="No numeric feature columns"):
        build_feature_matrix(df, feature_cols=["status"])

def test_build_engineered_features_adds_expected_columns():
    df = pd.DataFrame({"sensor_1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    
    result = build_engineered_features(df, feature_cols=["sensor_1"], rolling_window=3)

    assert "sensor_1_roll_mean" in result.columns
    assert "sensor_1_roll_std" in result.columns
    assert "sensor_1_diff" in result.columns
    assert "sensor_1" in result.columns

def test_build_engineered_features_no_nans():
    df = pd.DataFrame({"sensor_1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    
    result = build_engineered_features(df, feature_cols=["sensor_1"], rolling_window=3)

    assert not result.isnull().any().any()

def test_build_engineered_features_preserves_row_count():
    df = pd.DataFrame({"sensor_1": [1.0, 2.0, 3.0, 4.0, 5.0]})

    result = build_engineered_features(df, feature_cols=["sensor_1"], rolling_window=3)

    assert len(result) == len(df)