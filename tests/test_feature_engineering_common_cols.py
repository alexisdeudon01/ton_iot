#!/usr/bin/env python3
"""Tests for common engineered feature columns in CIC and TON datasets."""
import pandas as pd

from src.core.feature_engineering import engineer_cic, engineer_ton


def test_engineer_cic_adds_ratio_columns():
    """Ensure CIC engineering adds expected ratio columns."""
    df = pd.DataFrame({
        "Total Fwd Bytes": [100.0, 200.0],
        "Total Bwd Bytes": [50.0, 100.0],
        "Total Fwd Packets": [10.0, 20.0],
        "Total Bwd Packets": [5.0, 10.0],
        "Flow Duration": [2.0, 4.0],
    })

    engineered = engineer_cic(df)

    for column in [
        "Flow_Bytes_s",
        "Flow_Packets_s",
        "Avg_Packet_Size",
        "Traffic_Direction_Ratio",
        "dataset_source",
    ]:
        assert column in engineered.columns, f"Missing engineered column: {column}"

    assert engineered.loc[0, "Flow_Bytes_s"] == 150.0 / 2.0
    assert engineered.loc[0, "Flow_Packets_s"] == 15.0 / 2.0
    assert engineered.loc[0, "Avg_Packet_Size"] == 150.0 / 15.0
    assert engineered.loc[0, "Traffic_Direction_Ratio"] == 100.0 / 50.0
    assert engineered["dataset_source"].unique().tolist() == [0]


def test_engineer_ton_adds_ratio_columns():
    """Ensure TON engineering adds expected ratio columns."""
    df = pd.DataFrame({
        "bytes_in": [80.0, 160.0],
        "bytes_out": [20.0, 40.0],
        "pkts_in": [8.0, 16.0],
        "pkts_out": [2.0, 4.0],
        "duration": [4.0, 8.0],
    })

    engineered = engineer_ton(df)

    for column in [
        "Flow_Bytes_s",
        "Flow_Packets_s",
        "Avg_Packet_Size",
        "Traffic_Direction_Ratio",
        "dataset_source",
    ]:
        assert column in engineered.columns, f"Missing engineered column: {column}"

    assert engineered.loc[0, "Flow_Bytes_s"] == 100.0 / 4.0
    assert engineered.loc[0, "Flow_Packets_s"] == 10.0 / 4.0
    assert engineered.loc[0, "Avg_Packet_Size"] == 100.0 / 10.0
    assert engineered.loc[0, "Traffic_Direction_Ratio"] == 20.0 / 80.0
    assert engineered["dataset_source"].unique().tolist() == [1]
