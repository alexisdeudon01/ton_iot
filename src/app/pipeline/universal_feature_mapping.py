from __future__ import annotations

from typing import List, Dict

import polars as pl

UNIVERSAL_FEATURES: List[str] = [
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

RATIO_FEATURES: List[str] = [
    "fwd_packet_ratio",
    "fwd_byte_ratio",
    "flow_asymmetry",
    "header_ratio",
]

OUTLIER_FEATURES: List[str] = [
    feature for feature in UNIVERSAL_FEATURES if feature not in RATIO_FEATURES
]

CIC_REQUIRED_COLUMNS: List[str] = [
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

TON_REQUIRED_COLUMNS: List[str] = [
    "duration",
    "src_pkts",
    "dst_pkts",
    "src_bytes",
    "dst_bytes",
]


def cic_expressions(epsilon: float = 1e-6) -> List[pl.Expr]:
    fwd_pkts = pl.col("Total Fwd Packets")
    bwd_pkts = pl.col("Total Backward Packets")
    fwd_bytes = pl.col("Total Length of Fwd Packets")
    bwd_bytes = pl.col("Total Length of Bwd Packets")
    total_packets = fwd_pkts + bwd_pkts
    total_bytes = fwd_bytes + bwd_bytes
    header_total = pl.col("Fwd Header Length") + pl.col("Bwd Header Length")

    return [
        pl.col("Flow Duration").alias("flow_duration"),
        fwd_pkts.alias("total_fwd_packets"),
        bwd_pkts.alias("total_bwd_packets"),
        fwd_bytes.alias("total_fwd_bytes"),
        bwd_bytes.alias("total_bwd_bytes"),
        total_packets.alias("total_packets"),
        total_bytes.alias("total_bytes"),
        (total_bytes / (total_packets + 1)).alias("bytes_per_packet"),
        (fwd_pkts / (total_packets + 1)).alias("fwd_packet_ratio"),
        (fwd_bytes / (total_bytes + 1)).alias("fwd_byte_ratio"),
        pl.col("Flow Bytes/s").alias("bytes_per_second"),
        pl.col("Flow Packets/s").alias("packets_per_second"),
        pl.col("Average Packet Size").alias("avg_packet_size"),
        ((fwd_pkts - bwd_pkts).abs() / (total_packets + 1)).alias("flow_asymmetry"),
        (header_total / (total_bytes + header_total + epsilon)).alias("header_ratio"),
    ]


def ton_expressions(epsilon: float = 1e-6) -> List[pl.Expr]:
    duration = pl.col("duration")
    src_pkts = pl.col("src_pkts")
    dst_pkts = pl.col("dst_pkts")
    src_bytes = pl.col("src_bytes")
    dst_bytes = pl.col("dst_bytes")
    total_packets = src_pkts + dst_pkts
    total_bytes = src_bytes + dst_bytes

    return [
        (duration * 1e6).alias("flow_duration"),
        src_pkts.alias("total_fwd_packets"),
        dst_pkts.alias("total_bwd_packets"),
        src_bytes.alias("total_fwd_bytes"),
        dst_bytes.alias("total_bwd_bytes"),
        total_packets.alias("total_packets"),
        total_bytes.alias("total_bytes"),
        (total_bytes / (total_packets + 1)).alias("bytes_per_packet"),
        (src_pkts / (total_packets + 1)).alias("fwd_packet_ratio"),
        (src_bytes / (total_bytes + 1)).alias("fwd_byte_ratio"),
        (total_bytes / (duration + epsilon)).alias("bytes_per_second"),
        (total_packets / (duration + epsilon)).alias("packets_per_second"),
        (total_bytes / (total_packets + 1)).alias("avg_packet_size"),
        ((src_pkts - dst_pkts).abs() / (total_packets + 1)).alias("flow_asymmetry"),
        ((total_packets * 40) / (total_bytes + total_packets * 40 + epsilon)).alias("header_ratio"),
    ]


def mapping_rows() -> List[Dict[str, str]]:
    return [
        {
            "feature": "flow_duration",
            "cic_formula": "Flow Duration",
            "ton_formula": "duration * 1e6",
            "mapping_type": "direct",
            "unit": "microseconds",
            "description": "Duree totale du flux (normalisee en microsecondes)",
        },
        {
            "feature": "total_fwd_packets",
            "cic_formula": "Total Fwd Packets",
            "ton_formula": "src_pkts",
            "mapping_type": "direct",
            "unit": "count",
            "description": "Nombre total de paquets forward (source -> destination)",
        },
        {
            "feature": "total_bwd_packets",
            "cic_formula": "Total Backward Packets",
            "ton_formula": "dst_pkts",
            "mapping_type": "direct",
            "unit": "count",
            "description": "Nombre total de paquets backward (destination -> source)",
        },
        {
            "feature": "total_fwd_bytes",
            "cic_formula": "Total Length of Fwd Packets",
            "ton_formula": "src_bytes",
            "mapping_type": "direct",
            "unit": "bytes",
            "description": "Volume total de donnees forward",
        },
        {
            "feature": "total_bwd_bytes",
            "cic_formula": "Total Length of Bwd Packets",
            "ton_formula": "dst_bytes",
            "mapping_type": "direct",
            "unit": "bytes",
            "description": "Volume total de donnees backward",
        },
        {
            "feature": "total_packets",
            "cic_formula": "Total Fwd Packets + Total Backward Packets",
            "ton_formula": "src_pkts + dst_pkts",
            "mapping_type": "computed",
            "unit": "count",
            "description": "Nombre total de paquets dans le flux",
        },
        {
            "feature": "total_bytes",
            "cic_formula": "Total Length of Fwd Packets + Total Length of Bwd Packets",
            "ton_formula": "src_bytes + dst_bytes",
            "mapping_type": "computed",
            "unit": "bytes",
            "description": "Volume total de donnees echangees",
        },
        {
            "feature": "bytes_per_packet",
            "cic_formula": "(Total Length of Fwd Packets + Total Length of Bwd Packets) / (Total Fwd Packets + Total Backward Packets + 1)",
            "ton_formula": "(src_bytes + dst_bytes) / (src_pkts + dst_pkts + 1)",
            "mapping_type": "computed",
            "unit": "bytes/packet",
            "description": "Taille moyenne des paquets",
        },
        {
            "feature": "fwd_packet_ratio",
            "cic_formula": "Total Fwd Packets / (Total Fwd Packets + Total Backward Packets + 1)",
            "ton_formula": "src_pkts / (src_pkts + dst_pkts + 1)",
            "mapping_type": "computed",
            "unit": "ratio",
            "description": "Proportion de paquets forward",
        },
        {
            "feature": "fwd_byte_ratio",
            "cic_formula": "Total Length of Fwd Packets / (Total Length of Fwd Packets + Total Length of Bwd Packets + 1)",
            "ton_formula": "src_bytes / (src_bytes + dst_bytes + 1)",
            "mapping_type": "computed",
            "unit": "ratio",
            "description": "Proportion de bytes forward",
        },
        {
            "feature": "bytes_per_second",
            "cic_formula": "Flow Bytes/s",
            "ton_formula": "(src_bytes + dst_bytes) / (duration + 1e-6)",
            "mapping_type": "computed",
            "unit": "bytes/second",
            "description": "Debit moyen du flux",
        },
        {
            "feature": "packets_per_second",
            "cic_formula": "Flow Packets/s",
            "ton_formula": "(src_pkts + dst_pkts) / (duration + 1e-6)",
            "mapping_type": "computed",
            "unit": "packets/second",
            "description": "Cadence de paquets",
        },
        {
            "feature": "avg_packet_size",
            "cic_formula": "Average Packet Size",
            "ton_formula": "(src_bytes + dst_bytes) / (src_pkts + dst_pkts + 1)",
            "mapping_type": "computed",
            "unit": "bytes",
            "description": "Taille moyenne des paquets",
        },
        {
            "feature": "flow_asymmetry",
            "cic_formula": "abs(Total Fwd Packets - Total Backward Packets) / (Total Fwd Packets + Total Backward Packets + 1)",
            "ton_formula": "abs(src_pkts - dst_pkts) / (src_pkts + dst_pkts + 1)",
            "mapping_type": "computed",
            "unit": "ratio",
            "description": "Degre d'asymetrie du flux",
        },
        {
            "feature": "header_ratio",
            "cic_formula": "(Fwd Header Length + Bwd Header Length) / (Total Bytes + Headers + epsilon)",
            "ton_formula": "(total_packets * 40) / (total_bytes + total_packets * 40 + epsilon)",
            "mapping_type": "approximate",
            "unit": "ratio",
            "description": "Ratio overhead headers vs total",
        },
    ]
