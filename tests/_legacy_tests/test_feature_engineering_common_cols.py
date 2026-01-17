"""
Tests for feature engineering - common columns and dataset_source
"""
import sys
from pathlib import Path
import pytest
import pandas as pd

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.core.feature_engineering import engineer_cic, engineer_ton


def test_feature_engineering_common_cols():
    """Test that engineer_cic and engineer_ton add common columns including dataset_source"""
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

    assert expected_cols.issubset(cic_engineered.columns), \
        f"CIC engineered columns missing expected columns. Expected: {expected_cols}, Got: {set(cic_engineered.columns)}"
    assert expected_cols.issubset(ton_engineered.columns), \
        f"TON engineered columns missing expected columns. Expected: {expected_cols}, Got: {set(ton_engineered.columns)}"
    
    cic_dataset_source = cic_engineered["dataset_source"].unique().tolist()
    assert cic_dataset_source == [0], \
        f"CIC dataset_source should be [0] (got {cic_dataset_source})"
    
    ton_dataset_source = ton_engineered["dataset_source"].unique().tolist()
    assert ton_dataset_source == [1], \
        f"TON dataset_source should be [1] (got {ton_dataset_source})"
