"""
Data loader with automatic dataset detection and user prompt for samples.
Usage:
    from src.data.data_loader import DataLoader
    loader = DataLoader()
    cic_df, ton_df = loader.load_datasets()
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, config_path: str = "config.yaml", config: Optional[dict] = None):
        if config is None:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
        else:
            self.config = config

        self.paths = self.config.get("paths", {})
        datasets = self.config.get("datasets", {})

        self.raw_cic = Path(self.paths.get("raw_cic") or datasets.get("cic_ddos2019") or "data/raw/cic_ddos2019")
        self.raw_ton = Path(self.paths.get("raw_ton") or datasets.get("ton_iot") or "data/raw/ton_iot")
        self.sample_cic = Path(self.paths.get("sample_cic", "data/sample/cic_ddos2019_sample.csv"))
        self.sample_ton = Path(self.paths.get("sample_ton", "data/sample/ton_iot_sample.csv"))
        self.random_state = self.config.get("data", {}).get("random_state", 42)

    def _check_full_datasets(self) -> Tuple[bool, bool]:
        """Check if full datasets exist in data/raw/."""
        cic_exists = self.raw_cic.exists() and any(self.raw_cic.glob("*.csv"))
        ton_exists = self.raw_ton.exists() and any(self.raw_ton.glob("*.csv"))
        return cic_exists, ton_exists

    def _check_samples(self) -> Tuple[bool, bool]:
        """Check if sample files exist."""
        return self.sample_cic.exists(), self.sample_ton.exists()

    def _get_size_mb(self, path: Path) -> float:
        """Get file/directory size in MB."""
        if path.is_file():
            return path.stat().st_size / (1024 * 1024)
        if path.is_dir():
            total = sum(f.stat().st_size for f in path.rglob("*.csv"))
            return total / (1024 * 1024)
        return 0.0

    def _prompt_user_for_samples(self) -> bool:
        """Ask user if they want to use sample files."""
        logger.warning("Full datasets not found in data/raw/")
        logger.info("It appears you are using the ZIP submission version.")
        logger.info("For complete evaluation, download full datasets from Google Drive (recommended).")

        sample_cic_exists, sample_ton_exists = self._check_samples()
        if sample_cic_exists and sample_ton_exists:
            logger.info("Sample files available:")
            logger.info(f"- cic_ddos2019_sample.csv (~{self._get_size_mb(self.sample_cic):.0f} MB, 10% of original)")
            logger.info(f"- ton_iot_sample.csv (~{self._get_size_mb(self.sample_ton):.0f} MB, 10% of original)")
            logger.info("Would you like to use the pre-generated 10% sample files for testing? (y/n)")

            while True:
                response = input().strip().lower()
                if response in ["y", "yes"]:
                    return True
                if response in ["n", "no"]:
                    return False
                print("Please enter 'y' or 'n'")
        else:
            logger.error("Sample files not found either!")
            logger.error("Please ensure data/sample/ contains the sample CSV files.")
            return False

    def load_datasets(self, auto_use_samples: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load datasets with automatic detection.

        Args:
            auto_use_samples: If True, skip user prompt and use samples automatically
        Returns:
            Tuple of (cic_dataframe, ton_dataframe)
        """
        full_cic, full_ton = self._check_full_datasets()

        if full_cic and full_ton:
            logger.info("Full datasets detected in data/raw/")
            cic_df = self._load_full_dataset(self.raw_cic, "CIC-DDoS2019")
            ton_df = self._load_full_dataset(self.raw_ton, "TON_IoT")
            logger.info("Using FULL datasets for evaluation")
            return cic_df, ton_df

        if auto_use_samples:
            use_samples = True
        else:
            use_samples = self._prompt_user_for_samples()

        if use_samples:
            logger.info("Loading sample datasets...")
            cic_df = pd.read_csv(self.sample_cic, low_memory=False)
            ton_df = pd.read_csv(self.sample_ton, low_memory=False)
            logger.info(f"CIC-DDoS2019 sample: {len(cic_df):,} rows (10% of original)")
            logger.info(f"TON_IoT sample: {len(ton_df):,} rows (10% of original)")
            logger.info("Note: Results may vary slightly from full dataset evaluation")
            return cic_df, ton_df

        print()
        print("To use full datasets, please:")
        print("1. Download from Google Drive (link shared with supervisor)")
        print("2. Extract to data/raw/cic_ddos2019/ and data/raw/ton_iot/")
        print("3. Run the programme again")
        print()
        sys.exit(0)

    def _load_full_dataset(self, path: Path, name: str) -> pd.DataFrame:
        """Load all CSV files from a directory."""
        if path.is_file():
            df = pd.read_csv(path, low_memory=False)
            size_gb = self._get_size_mb(path) / 1024
            logger.info(f"Loading {name}: {size_gb:.1f} GB ({len(df):,} rows)")
            return df

        csv_files = list(path.glob("*.csv"))
        dfs = []
        for f in csv_files:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        size_gb = self._get_size_mb(path) / 1024
        logger.info(f"Loading {name}: {size_gb:.1f} GB ({len(combined):,} rows)")
        return combined


def load_data(auto_samples: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Quick function to load datasets."""
    loader = DataLoader()
    return loader.load_datasets(auto_use_samples=auto_samples)


if __name__ == "__main__":
    cic, ton = load_data()
    print(f"\nLoaded CIC-DDoS2019: {len(cic):,} rows")
    print(f"Loaded TON_IoT: {len(ton):,} rows")
