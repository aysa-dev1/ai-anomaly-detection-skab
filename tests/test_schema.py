import pandas as pd

from anomaly_detection.data.schema import detect_label_column, detect_timestamp_column


def test_detect_columns_case_insensitive():
    df = pd.DataFrame(columns=["TimeStamp", "Anomaly", "sensor_1"])
    assert detect_timestamp_column(df, ["timestamp"]) == "TimeStamp"
    assert detect_label_column(df, ["anomaly"]) == "Anomaly"
