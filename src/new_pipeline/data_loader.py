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
    """Phase 1: Dask-based Data Loading with Out-of-Core Support and Harmonization"""

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
        """Loads ToN-IoT and CICDDoS2019 datasets using Dask and harmonizes features."""
        print("\n" + "=" * 80)
        print(f"MICRO-TÂCHE: Chargement et Harmonisation via DASK")
        print(f"STRATÉGIE: Mapping Sémantique, Dtypes Robustes, Out-of-Core")
        print("=" * 80)

        start_time = time.time()

        # 1. Define Semantic Mapping (Ensuring at least 4 categories)
        # Unified Name -> {ton: name, cic: name}
        mapping = {
            # Category: Flow Identifiers
            "src_ip": {"ton": "src_ip", "cic": "Source IP"},
            "src_port": {"ton": "src_port", "cic": "Source Port"},
            "dst_ip": {"ton": "dst_ip", "cic": "Destination IP"},
            "dst_port": {"ton": "dst_port", "cic": "Destination Port"},
            "proto": {"ton": "proto", "cic": "Protocol"},
            # Category: Flow Basic Stats
            "duration": {"ton": "duration", "cic": "Flow Duration"},
            "fwd_pkts": {"ton": "src_pkts", "cic": "Total Fwd Packets"},
            "bwd_pkts": {"ton": "dst_pkts", "cic": "Total Backward Packets"},
            "fwd_bytes": {"ton": "src_bytes", "cic": "Total Length of Fwd Packets"},
            "bwd_bytes": {"ton": "dst_bytes", "cic": "Total Length of Bwd Packets"},
            # Category: Packet Length Stats
            "fwd_pkt_len_max": {"ton": "fwd_pkt_len_max", "cic": "Fwd Packet Length Max"},
            "fwd_pkt_len_min": {"ton": "fwd_pkt_len_min", "cic": "Fwd Packet Length Min"},
            "bwd_pkt_len_max": {"ton": "bwd_pkt_len_max", "cic": "Bwd Packet Length Max"},
            "bwd_pkt_len_min": {"ton": "bwd_pkt_len_min", "cic": "Bwd Packet Length Min"},
            # Category: Throughput Stats
            "flow_byts_s": {"ton": "flow_byts_s", "cic": "Flow Bytes/s"},
            "flow_pkts_s": {"ton": "flow_pkts_s", "cic": "Flow Packets/s"},
        }

        # 2. Load ToN-IoT (Lazy)
        ton_ddf = None
        if ton_iot_path.exists():
            print(f"INFO: Préparation du chargement de ToN-IoT: {ton_iot_path.name}")
            ton_ddf = dd.read_csv(ton_iot_path, low_memory=False, assume_missing=True)
            ton_ddf.columns = [c.strip() for c in ton_ddf.columns]

            # Apply mapping
            ton_rename = {v["ton"]: k for k, v in mapping.items() if v["ton"] in ton_ddf.columns}
            ton_ddf = ton_ddf.rename(columns=ton_rename)

            if "type" in ton_ddf.columns:
                ton_ddf = ton_ddf[ton_ddf["type"].isin(["normal", "ddos"])]
                ton_ddf["is_ddos"] = ton_ddf["type"].map(
                    {"normal": 0, "ddos": 1}, meta=("is_ddos", "i8")
                )
                ton_ddf["type"] = ton_ddf["is_ddos"]
            ton_ddf["dataset"] = "ton_iot"

        # 3. Load CICDDoS2019 (Lazy & Recursive)
        cic_ddf = None
        if cic_ddos_dir.exists():
            cic_pattern = str(cic_ddos_dir / "**" / "*.csv")
            print(f"INFO: Recherche récursive des fichiers CICDDoS2019: {cic_pattern}")

            cic_ddf = dd.read_csv(
                cic_pattern,
                low_memory=False,
                assume_missing=True,
                dtype='object' # Load as object to avoid inference issues
            )
            cic_ddf.columns = [c.strip() for c in cic_ddf.columns]

            # Apply mapping
            cic_rename = {v["cic"]: k for k, v in mapping.items() if v["cic"] in cic_ddf.columns}
            cic_ddf = cic_ddf.rename(columns=cic_rename)

            if "Label" in cic_ddf.columns:
                cic_ddf["is_ddos"] = (
                    cic_ddf["Label"].astype(str).str.upper() != "BENIGN"
                ).astype(int)
                cic_ddf["type"] = cic_ddf["is_ddos"]
            cic_ddf["dataset"] = "cic_ddos2019"

        # 4. Combine Datasets
        if ton_ddf is not None and cic_ddf is not None:
            cols_to_keep = list(mapping.keys()) + ["is_ddos", "type", "dataset", "ts", "Timestamp"]
            final_cols = [c for c in cols_to_keep if c in ton_ddf.columns and c in cic_ddf.columns]

            print(f"INFO: Fusion de {len(final_cols)} colonnes harmonisées.")
            self.ddf = dd.concat([ton_ddf[final_cols], cic_ddf[final_cols]])
        elif ton_ddf is not None:
            self.ddf = ton_ddf
        elif cic_ddf is not None:
            self.ddf = cic_ddf
        else:
            raise FileNotFoundError("Aucun fichier de données trouvé.")

        # Convert numeric columns
        numeric_cols = [k for k, v in mapping.items() if k not in ["src_ip", "dst_ip"]]
        for col in numeric_cols:
            if col in self.ddf.columns:
                self.ddf[col] = dd.to_numeric(self.ddf[col], errors='coerce').fillna(0)

        if sample_ratio < 1.0:
            self.ddf = self.ddf.sample(frac=sample_ratio, random_state=42)

        print(f"RÉSULTAT: Graphe Dask construit. Temps: {time.time() - start_time:.2f}s")

        # Save consolidated dataset to rr/ as requested
        consolidated_path = self.rr_dir / "consolidated_dataset.parquet"
        self.ddf.to_parquet(consolidated_path, write_index=False)
        print(f"RÉSULTAT: Dataset consolidé sauvegardé dans {consolidated_path}")

        return self.ddf

    def profile_and_validate(self):
        """Profiling using Dask compute for necessary statistics."""
        if self.ddf is None:
            raise ValueError("Load data first.")

        print("\n" + "-" * 40)
        print("MICRO-TÂCHE: Validation et Profilage (Calcul Dask)")

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

        try:
            # Use sample to ensure class diversity
            return self.ddf.sample(frac=min(1.0, (n_rows*5)/70000000)).compute().head(n_rows)
        except:
            return self.ddf.head(n_rows)
