import pandas as pd
import pytest

from anomaly_detection.data.split import time_series_split


def test_time_series_split_basic():
    df = pd.DataFrame({"value": range(10)})

    train, test = time_series_split(df, train_ratio=0.7)

    assert len(train) == 7
    assert len(test) == 3

    # ensure chronological order preserved
    assert train.iloc[-1]["value"] == 6
    assert test.iloc[0]["value"] == 7


def test_time_series_split_invalid_ratio():
    df = pd.DataFrame({"value": range(5)})

    with pytest.raises(ValueError):
        time_series_split(df, train_ratio=1.2)


def test_time_series_split_too_small():
    df = pd.DataFrame({"value": [1]})

    with pytest.raises(ValueError):
        time_series_split(df, train_ratio=0.7)
