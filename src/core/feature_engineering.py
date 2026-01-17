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
    Handles potential leading/trailing spaces in column names.

    Args:
        df: Input dataframe with CIC columns.
        eps: Small value to avoid division by zero.

    Returns:
        Dataframe with engineered columns added.
    """
    engineered = df.copy()

    # Clean column names (remove leading/trailing spaces)
    engineered.columns = [col.strip() for col in engineered.columns]

    # Expected columns after stripping:
    # "Total Fwd Bytes", "Total Bwd Bytes", "Total Fwd Packets", "Total Bwd Packets", "Flow Duration"

    try:
        total_bytes = engineered["Total Fwd Bytes"] + engineered["Total Bwd Bytes"]
        total_packets = engineered["Total Fwd Packets"] + engineered["Total Bwd Packets"]
        duration = engineered["Flow Duration"]

        # Use max(duration, eps) to avoid division by zero and ensure scientific correctness
        safe_duration = duration.clip(lower=eps)
        safe_packets = total_packets.clip(lower=eps)

        engineered["Flow_Bytes_s"] = total_bytes / safe_duration
        engineered["Flow_Packets_s"] = total_packets / safe_duration
        engineered["Avg_Packet_Size"] = total_bytes / safe_packets
        engineered["Traffic_Direction_Ratio"] = engineered["Total Fwd Bytes"] / (
            engineered["Total Bwd Bytes"] + eps
        )
    except KeyError as e:
        logger.warning(f"Missing expected CIC columns for feature engineering: {e}")
        # Fallback or skip ratios if columns missing
        pass

    engineered["dataset_source"] = 0

    logger.info("Engineered CIC flow ratios and dataset source.")
    return engineered


def engineer_ton(df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    """
    Compute flow ratios for TON_IoT data.
    Handles both 'bytes_in/out' and 'src/dst_bytes' naming conventions.

    Args:
        df: Input dataframe with TON columns.
        eps: Small value to avoid division by zero.

    Returns:
        Dataframe with engineered columns added.
    """
    engineered = df.copy()

    # Handle different naming conventions for TON_IoT
    b_in = "bytes_in" if "bytes_in" in engineered.columns else "src_bytes"
    b_out = "bytes_out" if "bytes_out" in engineered.columns else "dst_bytes"
    p_in = "pkts_in" if "pkts_in" in engineered.columns else "src_pkts"
    p_out = "pkts_out" if "pkts_out" in engineered.columns else "dst_pkts"

    total_bytes = engineered[b_in] + engineered[b_out]
    total_packets = engineered[p_in] + engineered[p_out]
    duration = engineered["duration"]

    # Use max(duration, eps) to avoid division by zero and ensure scientific correctness
    safe_duration = duration.clip(lower=eps)
    safe_packets = total_packets.clip(lower=eps)

    engineered["Flow_Bytes_s"] = total_bytes / safe_duration
    engineered["Flow_Packets_s"] = total_packets / safe_duration
    engineered["Avg_Packet_Size"] = total_bytes / safe_packets
    engineered["Traffic_Direction_Ratio"] = engineered[b_out] / (
        engineered[b_in] + eps
    )
    engineered["dataset_source"] = 1

    logger.info(f"Engineered TON flow ratios (using {b_in}/{b_out}) and dataset source.")
    return engineered
