from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

UNIVERSAL_FEATURES = [
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "total_packets",
    "total_bytes",
    "bytes_per_packet",
    "fwd_packet_ratio",
    "fwd_byte_ratio",
    "bytes_per_second",
    "packets_per_second",
    "avg_packet_size",
    "flow_asymmetry",
    "header_ratio",
]


class UniversalFeatureEngineer:
    """Compute universal features for CIC-DDoS2019 and TON_IoT datasets."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon

    def _require_columns(self, df: pd.DataFrame, columns: Iterable[str], dataset: str) -> None:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in {dataset}: {', '.join(missing)}")

    @staticmethod
    def _to_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        for col in cleaned.columns:
            series = self._to_numeric(cleaned[col]).replace([np.inf, -np.inf], np.nan)
            finite = series.dropna()
            max_val = finite.max() if not finite.empty else 0.0
            series = series.fillna(max_val).replace([np.inf, -np.inf], max_val).fillna(0.0)
            cleaned[col] = series
        return cleaned

    def transform_cic(self, df: pd.DataFrame) -> pd.DataFrame:
        required = [
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Fwd Header Length",
            "Bwd Header Length",
            "Flow Bytes/s",
            "Flow Packets/s",
            "Average Packet Size",
        ]
        self._require_columns(df, required, "CIC-DDoS2019")

        fwd_pkts = self._to_numeric(df["Total Fwd Packets"])
        bwd_pkts = self._to_numeric(df["Total Backward Packets"])
        fwd_bytes = self._to_numeric(df["Total Length of Fwd Packets"])
        bwd_bytes = self._to_numeric(df["Total Length of Bwd Packets"])
        total_packets = fwd_pkts + bwd_pkts
        total_bytes = fwd_bytes + bwd_bytes
        header_total = self._to_numeric(df["Fwd Header Length"]) + self._to_numeric(
            df["Bwd Header Length"]
        )

        features = pd.DataFrame(
            {
                "flow_duration": self._to_numeric(df["Flow Duration"]),
                "total_fwd_packets": fwd_pkts,
                "total_bwd_packets": bwd_pkts,
                "total_fwd_bytes": fwd_bytes,
                "total_bwd_bytes": bwd_bytes,
                "total_packets": total_packets,
                "total_bytes": total_bytes,
                "bytes_per_packet": total_bytes / (total_packets + 1),
                "fwd_packet_ratio": fwd_pkts / (total_packets + 1),
                "fwd_byte_ratio": fwd_bytes / (total_bytes + 1),
                "bytes_per_second": self._to_numeric(df["Flow Bytes/s"]),
                "packets_per_second": self._to_numeric(df["Flow Packets/s"]),
                "avg_packet_size": self._to_numeric(df["Average Packet Size"]),
                "flow_asymmetry": (fwd_pkts - bwd_pkts).abs() / (total_packets + 1),
                "header_ratio": header_total / (total_bytes + header_total + self.epsilon),
            }
        )

        return self._clean_dataframe(features)[UNIVERSAL_FEATURES]

    def transform_ton(self, df: pd.DataFrame) -> pd.DataFrame:
        required = [
            "duration",
            "src_pkts",
            "dst_pkts",
            "src_bytes",
            "dst_bytes",
        ]
        self._require_columns(df, required, "TON_IoT")

        duration = self._to_numeric(df["duration"])
        src_pkts = self._to_numeric(df["src_pkts"])
        dst_pkts = self._to_numeric(df["dst_pkts"])
        src_bytes = self._to_numeric(df["src_bytes"])
        dst_bytes = self._to_numeric(df["dst_bytes"])
        total_packets = src_pkts + dst_pkts
        total_bytes = src_bytes + dst_bytes

        features = pd.DataFrame(
            {
                "flow_duration": duration * 1e6,
                "total_fwd_packets": src_pkts,
                "total_bwd_packets": dst_pkts,
                "total_fwd_bytes": src_bytes,
                "total_bwd_bytes": dst_bytes,
                "total_packets": total_packets,
                "total_bytes": total_bytes,
                "bytes_per_packet": total_bytes / (total_packets + 1),
                "fwd_packet_ratio": src_pkts / (total_packets + 1),
                "fwd_byte_ratio": src_bytes / (total_bytes + 1),
                "bytes_per_second": total_bytes / (duration + self.epsilon),
                "packets_per_second": total_packets / (duration + self.epsilon),
                "avg_packet_size": total_bytes / (total_packets + 1),
                "flow_asymmetry": (src_pkts - dst_pkts).abs() / (total_packets + 1),
                "header_ratio": (total_packets * 40) / (total_bytes + total_packets * 40 + self.epsilon),
            }
        )

        return self._clean_dataframe(features)[UNIVERSAL_FEATURES]
