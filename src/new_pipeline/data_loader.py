import pandas as pd
import numpy as np
import logging
import time
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, Optional, List
from src.system_monitor import SystemMonitor
import psutil
import gc

logger = logging.getLogger(__name__)

class RealDataLoader:
    """Phase 1: Multi-threaded Data Loading with Resource Awareness and Filtering"""

    def __init__(self, monitor: SystemMonitor, target_col: str = 'is_ddos', rr_dir: Path = Path("rr")):
        self.monitor = monitor
        self.target_col = target_col
        self.rr_dir = rr_dir
        self.df: Optional[pd.DataFrame] = None
        self.splits: Optional[Dict[str, pd.DataFrame]] = None

    def load_datasets(self, ton_iot_path: Path, cic_ddos_dir: Path, sample_ratio: float = 1.0) -> pd.DataFrame:
        """Loads ToN-IoT and CICDDoS2019 datasets with specific filtering and mapping."""
        print("\n" + "="*80)
        print(f"MICRO-TÂCHE: Chargement des datasets (ToN-IoT & CICDDoS2019)")
        print(f"STRATÉGIE: Multi-threading, Filtrage ToN-IoT, Mapping CICDDoS2019, Limite RAM 50%")
        print("="*80)

        all_dfs = []
        start_time = time.time()

        # 1. Identify all files
        tasks = []
        if ton_iot_path.exists():
            tasks.append(('ton_iot', ton_iot_path))

        if cic_ddos_dir.exists():
            # Recursive search for CSV files in CICDDoS2019 directory
            cic_files = list(cic_ddos_dir.rglob("*.csv"))
            for f in cic_files:
                tasks.append(('cic_ddos', f))

        if not tasks:
            raise FileNotFoundError("Aucun fichier de données trouvé.")

        # 2. Load files using ThreadPoolExecutor
        max_workers = min(len(tasks), (psutil.cpu_count() or 2))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(self._load_single_file, source, path, sample_ratio): (source, path) for source, path in tasks}

            for future in as_completed(future_to_task):
                source, path = future_to_task[future]
                try:
                    df_part = future.result()
                    if df_part is not None and not df_part.empty:
                        all_dfs.append(df_part)
                        # Verbose output for each dataset part
                        self._print_dataset_info(source, path, df_part)
                except Exception as exc:
                    print(f"ERREUR: {path.name} a généré une exception: {exc}")

        if not all_dfs:
            raise ValueError("Aucune donnée n'a pu être chargée.")

        print("\n" + "-"*40)
        print("MICRO-TÂCHE: Consolidation des données")
        self.df = pd.concat(all_dfs, ignore_index=True)

        # Final type conversion for consistency
        if 'type' in self.df.columns:
            # Ensure 'type' is binary as requested: 0 if not ddos, 1 if ddos
            # For ToN-IoT, we already filtered to only have 'normal' and 'ddos'
            # For CICDDoS, we mapped it during loading
            pass

        print(f"RÉSULTAT: Dataset consolidé. Shape: {self.df.shape}. Temps total: {time.time() - start_time:.2f}s")

        # Save consolidated Parquet (more efficient than CSV)
        consolidated_path = Path("output/consolidated_data.parquet")
        consolidated_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove 'dataset' column before saving to parquet as requested
        df_to_save = self.df.drop(columns=['dataset'], errors='ignore')
        df_to_save.to_parquet(consolidated_path, index=False)
        print(f"RÉSULTAT: Fichier Parquet consolidé sauvegardé dans {consolidated_path}")

        return self.df

    def _load_single_file(self, source: str, file_path: Path, sample_ratio: float) -> Optional[pd.DataFrame]:
        """Loads a single file with specific logic for ToN-IoT and CICDDoS2019."""
        # Wait if RAM is too high (Dedicated Resource Management)
        while self.monitor.get_memory_info()['used_percent'] > self.monitor.max_memory_percent:
            gc.collect()
            time.sleep(1.0)

        try:
            if source == 'ton_iot':
                # ToN-IoT: Only keep 'normal' or 'ddos' in 'type' column
                df = pd.read_csv(file_path, low_memory=False)
                if 'type' in df.columns:
                    df = df[df['type'].isin(['normal', 'ddos'])].copy()
                    # Map type to binary: 0 if normal, 1 if ddos
                    df['type'] = df['type'].map({'normal': 0, 'ddos': 1})
                    df['is_ddos'] = df['type']
                df['dataset'] = 'ton_iot'

            elif source == 'cic_ddos':
                # CICDDoS2019: Map 'Label' BENIGN -> 0, others -> 1
                df = pd.read_csv(file_path, low_memory=False)
                if 'Label' in df.columns:
                    df['is_ddos'] = (df['Label'].astype(str).str.upper() != 'BENIGN').astype(int)
                    # Also update 'type' column to be binary
                    df['type'] = df['is_ddos']
                df['dataset'] = 'cic_ddos2019'

            else:
                return None

            if sample_ratio < 1.0:
                df = df.sample(frac=sample_ratio, random_state=42)

            # Optimize memory immediately
            df = self._optimize_dtypes(df)
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None

    def _print_dataset_info(self, source: str, path: Path, df: pd.DataFrame):
        """Prints header and a random row for the dataset."""
        print(f"\nINFO DATASET: {source} | Fichier: {path.name}")
        print(f"HEADER: {list(df.columns)}")
        if len(df) > 1:
            random_idx = random.randint(0, len(df) - 1)
            print(f"VALEUR ALÉATOIRE (Ligne {random_idx}):")
            print(df.iloc[random_idx].to_dict())
        print("-" * 20)

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduces memory usage by downcasting types."""
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        return df

    def profile_and_validate(self) -> dict:
        """Profiling and Class Distribution Plot."""
        if self.df is None:
            raise ValueError("Load data first.")

        total_rows = len(self.df)
        counts = self.df['is_ddos'].value_counts()

        # Splits
        train_df, temp_df = train_test_split(self.df, test_size=0.4, stratify=self.df['is_ddos'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['is_ddos'], random_state=42)
        self.splits = {'train': train_df, 'val': val_df, 'test': test_df}

        # Plots
        self._plot_distributions(counts)

        return {
            'total_rows': total_rows,
            'split_counts': {k: len(v) for k, v in self.splits.items()}
        }

    def _plot_distributions(self, counts):
        plt.figure(figsize=(8, 6))
        sns.barplot(x=['Normal/Benign', 'DDoS'], y=[counts.get(0, 0), counts.get(1, 0)])
        plt.title("Class Distribution (ToN-IoT & CICDDoS2019)")
        plt.savefig(self.rr_dir / "phase1_distribution.png")
        plt.close()

    def get_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.splits is None:
            raise ValueError("Splits not generated.")
        return self.splits['train'], self.splits['val'], self.splits['test']
