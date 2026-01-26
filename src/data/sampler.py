"""
Script to generate 10% stratified samples from full datasets.
Run this once after downloading full datasets from Google Drive.
Usage: python -m src.data.sampler
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SAMPLE_RATIO = 0.10
RANDOM_STATE = 42


def _detect_label_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["Label", "label", "y", "type", "attack", "class"]:
        if col in df.columns:
            return col
    return None


def create_sample(input_dir: Path, output_file: Path, label_col: str | None = None) -> None:
    """Create a 10% stratified sample from CSV files in a directory."""
    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {input_dir}")
        return

    logger.info(f"Loading {len(csv_files)} files from {input_dir}...")
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)
        logger.info(f" Loaded {f.name}: {len(df):,} rows")

    full_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total rows: {len(full_df):,}")

    label = label_col or _detect_label_column(full_df)
    if label and label in full_df.columns:
        _, sample_df = train_test_split(
            full_df,
            test_size=SAMPLE_RATIO,
            stratify=full_df[label],
            random_state=RANDOM_STATE,
        )
        logger.info(f"Stratified sampling on '{label}' column")
    else:
        _, sample_df = train_test_split(
            full_df,
            test_size=SAMPLE_RATIO,
            random_state=RANDOM_STATE,
        )
        logger.warning("Label column not found, using random sampling")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(output_file, index=False)
    size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"Saved: {output_file} ({size_mb:.0f} MB, {len(sample_df):,} rows)")


def main() -> None:
    print("=" * 60)
    print("DATASET SAMPLING (10%)")
    print("=" * 60)
    print()

    cic_raw = Path("data/raw/cic_ddos2019")
    cic_sample = Path("data/sample/cic_ddos2019_sample.csv")
    if cic_raw.exists():
        create_sample(cic_raw, cic_sample, label_col="Label")
    else:
        logger.warning("CIC-DDoS2019 full dataset not found in data/raw/cic_ddos2019")

    ton_raw = Path("data/raw/ton_iot")
    ton_sample = Path("data/sample/ton_iot_sample.csv")
    if ton_raw.exists():
        create_sample(ton_raw, ton_sample, label_col="type")
    else:
        logger.warning("TON_IoT full dataset not found in data/raw/ton_iot")


if __name__ == "__main__":
    main()
