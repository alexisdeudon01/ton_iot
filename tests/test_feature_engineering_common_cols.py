import pandas as pd

from src.core.feature_engineering import engineer_cic, engineer_ton


def test_feature_engineering_common_cols():
    cic_df = pd.DataFrame(
        {
            "Total Fwd Bytes": [100, 200],
            "Total Bwd Bytes": [50, 100],
            "Total Fwd Packets": [10, 20],
            "Total Bwd Packets": [5, 10],
            "Flow Duration": [2, 4],
            "label": [0, 1],
        }
    )
    ton_df = pd.DataFrame(
        {
            "bytes_in": [300, 400],
            "bytes_out": [150, 200],
            "pkts_in": [30, 40],
            "pkts_out": [15, 20],
            "duration": [3, 6],
            "label": [0, 1],
        }
    )

    cic_engineered = engineer_cic(cic_df)
    ton_engineered = engineer_ton(ton_df)

    expected_cols = {
        "Flow_Bytes_s",
        "Flow_Packets_s",
        "Avg_Packet_Size",
        "Traffic_Direction_Ratio",
        "dataset_source",
    }

    assert expected_cols.issubset(cic_engineered.columns)
    assert expected_cols.issubset(ton_engineered.columns)
    assert cic_engineered["dataset_source"].unique().tolist() == [0]
    assert ton_engineered["dataset_source"].unique().tolist() == [1]
