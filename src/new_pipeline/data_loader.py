import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dask.distributed import Client
from sklearn.model_selection import train_test_split

from src.system_monitor import SystemMonitor

logger = logging.getLogger(__name__)


class RealDataLoader:
    """Phase 1: Dask-based Data Loading with Out-of-Core Support"""

    def __init__(
        self, monitor: SystemMonitor, target_col: str = "is_ddos", rr_dir: Path = Path("rr")
    ):
        self.monitor = monitor
        self.target_col = target_col
        self.rr_dir = rr_dir
        self.ddf: Optional[dd.DataFrame] = None
        self.splits: Optional[Dict[str, Any]] = None

    def load_datasets(
        self, ton_iot_path: Path, cic_ddos_dir: Path, sample_ratio: float = 1.0
    ) -> Optional[dd.DataFrame]:
        """Loads ToN-IoT and CICDDoS2019 datasets using Dask for memory efficiency."""
        print("\n" + "=" * 80)
        print(f"MICRO-TÂCHE: Chargement des datasets via DASK (Out-of-Core)")
        print(f"STRATÉGIE: Chargement Lazy, Filtrage Dask, Mapping Distribué")
        print("=" * 80)

        start_time = time.time()

        # 1. Load ToN-IoT (Lazy)
        ton_ddf = None
        if ton_iot_path.exists():
            print(f"INFO: Préparation du chargement de ToN-IoT: {ton_iot_path.name}")
            # Use assume_missing=True to handle mixed types and NaNs gracefully
            ton_ddf = dd.read_csv(ton_iot_path, low_memory=False, assume_missing=True)
            # Clean column names (strip spaces)
            ton_ddf.columns = [c.strip() for c in ton_ddf.columns]

            # Filtering and mapping (Lazy)
            if "type" in ton_ddf.columns:
                ton_ddf = ton_ddf[ton_ddf["type"].isin(["normal", "ddos"])]
                ton_ddf["type"] = ton_ddf["type"].map(
                    {"normal": 0, "ddos": 1}, meta=("type", "i8")
                )
                ton_ddf["is_ddos"] = ton_ddf["type"]
            ton_ddf["dataset"] = "ton_iot"

        # 2. Load CICDDoS2019 (Lazy & Recursive)
        cic_ddf = None
        if cic_ddos_dir.exists():
            cic_pattern = str(cic_ddos_dir / "**" / "*.csv")
            print(f"INFO: Recherche récursive des fichiers CICDDoS2019: {cic_pattern}")

            # Dask handles recursive globbing. We use assume_missing=True and explicit object for problematic cols.
            cic_ddf = dd.read_csv(
                cic_pattern,
                low_memory=False,
                assume_missing=True,
                dtype={
                    'SimillarHTTP': 'object',
                    'Flow ID': 'object',
                    'Source IP': 'object',
                    'Destination IP': 'object',
                    'Timestamp': 'object',
                    ' Label': 'object' # Note the leading space often found in CIC datasets
                }
            )
            # Clean column names immediately
            cic_ddf.columns = [c.strip() for c in cic_ddf.columns]

            if "Label" in cic_ddf.columns:
                # Map Label: BENIGN -> 0, others -> 1
                cic_ddf["is_ddos"] = (
                    cic_ddf["Label"].astype(str).str.upper() != "BENIGN"
                ).astype(int)
                cic_ddf["type"] = cic_ddf["is_ddos"]
            cic_ddf["dataset"] = "cic_ddos2019"

        # 3. Combine Datasets
        if ton_ddf is not None and cic_ddf is not None:
            # Align columns before concat
            ton_cols = set(ton_ddf.columns)
            cic_cols = set(cic_ddf.columns)
            common_features = list(ton_cols & cic_cols)

            # Ensure our target columns are included
            for col in ["is_ddos", "type", "dataset"]:
                if col not in common_features:
                    # If missing from one, we can't easily concat without alignment
                    # But here they should be in both as we just created them
                    common_features.append(col)

            print(f"INFO: Fusion de {len(common_features)} colonnes communes.")
            self.ddf = dd.concat([ton_ddf[common_features], cic_ddf[common_features]])
        elif ton_ddf is not None:
            self.ddf = ton_ddf
        elif cic_ddf is not None:
            self.ddf = cic_ddf
        else:
            raise FileNotFoundError("Aucun fichier de données trouvé.")

        if self.ddf is not None and sample_ratio < 1.0:
            self.ddf = self.ddf.sample(frac=sample_ratio, random_state=42)

        print(
            f"RÉSULTAT: Graphe Dask construit. Temps: {time.time() - start_time:.2f}s"
        )
        return self.ddf

    def profile_and_validate(self):
        """Profiling using Dask compute for necessary statistics."""
        if self.ddf is None:
            raise ValueError("Load data first.")

        print("\n" + "-" * 40)
        print("MICRO-TÂCHE: Validation et Profilage (Calcul Dask)")

        # Compute only what's needed
        counts = self.ddf["is_ddos"].value_counts().compute()
        total_rows = counts.sum()

        print(f"RÉSULTAT: Dataset prêt. Lignes totales: {total_rows}")
        print(f"DISTRIBUTION:\n{counts}")

        # Splits (Dask-based split)
        train, test = self.ddf.random_split([0.8, 0.2], random_state=42)
        val, test = test.random_split([0.5, 0.5], random_state=42)

        self.splits = {"train": train, "val": val, "test": test}

        self._plot_distributions(counts)

    def _plot_distributions(self, counts):
        plt.figure(figsize=(8, 6))
        sns.barplot(x=["Normal/Benign", "DDoS"], y=[counts.get(0, 0), counts.get(1, 0)])
        plt.title("Class Distribution (Dask Loaded)")
        plt.savefig(self.rr_dir / "phase1_distribution.png")
        plt.close()

    def get_splits(self) -> Tuple[Any, Any, Any]:
        if self.splits is None:
            raise ValueError("Splits not generated.")
        return self.splits["train"], self.splits["val"], self.splits["test"]

    def get_sample_pandas(self, n_rows: int = 10000) -> pd.DataFrame:
        """Returns a small pandas sample for visualizations or non-Dask models."""
        if self.ddf is None:
            return pd.DataFrame()
        return self.ddf.head(n_rows)
