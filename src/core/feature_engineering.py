"""
Feature engineering helpers for CIC-DDoS2019 and TON_IoT datasets.

Expected columns:
- CIC-DDoS2019: "Total Fwd Bytes", "Total Bwd Bytes", "Total Fwd Packets",
  "Total Bwd Packets", "Flow Duration".
- TON_IoT: "bytes_in", "bytes_out", "pkts_in", "pkts_out", "duration".
"""
from __future__ import annotations

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def engineer_cic(df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    """
    Compute flow ratios for CIC-DDoS2019 data.

    Args:
        df: Input dataframe with CIC columns.
        eps: Small value to avoid division by zero.

    Returns:
        Dataframe with engineered columns added.
    """
    engineered = df.copy()
    total_bytes = engineered["Total Fwd Bytes"] + engineered["Total Bwd Bytes"]
    total_packets = engineered["Total Fwd Packets"] + engineered["Total Bwd Packets"]
    duration = engineered["Flow Duration"]

    engineered["Flow_Bytes_s"] = total_bytes / (duration + eps)
    engineered["Flow_Packets_s"] = total_packets / (duration + eps)
    engineered["Avg_Packet_Size"] = total_bytes / (total_packets + eps)
    engineered["Traffic_Direction_Ratio"] = engineered["Total Fwd Bytes"] / (
        engineered["Total Bwd Bytes"] + eps
    )
    engineered["dataset_source"] = 0

    logger.info("Engineered CIC flow ratios and dataset source.")
    return engineered


def engineer_ton(df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    """
    Compute flow ratios for TON_IoT data.

    Args:
        df: Input dataframe with TON columns.
        eps: Small value to avoid division by zero.

    Returns:
        Dataframe with engineered columns added.
    """
    engineered = df.copy()
    total_bytes = engineered["bytes_in"] + engineered["bytes_out"]
    total_packets = engineered["pkts_in"] + engineered["pkts_out"]
    duration = engineered["duration"]

    engineered["Flow_Bytes_s"] = total_bytes / (duration + eps)
    engineered["Flow_Packets_s"] = total_packets / (duration + eps)
    engineered["Avg_Packet_Size"] = total_bytes / (total_packets + eps)
    engineered["Traffic_Direction_Ratio"] = engineered["bytes_out"] / (
        engineered["bytes_in"] + eps
    )
    engineered["dataset_source"] = 1

    logger.info("Engineered TON flow ratios and dataset source.")
    return engineered
