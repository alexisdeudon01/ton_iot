import pandas as pd
import numpy as np
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, Optional, List
from src.system_monitor import SystemMonitor
import psutil

logger = logging.getLogger(__name__)

class RealDataLoader:
    """Phase 1: Multi-threaded Data Loading with Resource Awareness"""

    def __init__(self, data_dir: Path, monitor: SystemMonitor, target_col: str = 'label', rr_dir: Path = Path("rr")):
        self.data_dir = data_dir
        self.monitor = monitor
        self.target_col = target_col
        self.rr_dir = rr_dir
        self.df: Optional[pd.DataFrame] = None
        self.splits: Optional[Dict[str, pd.DataFrame]] = None

    def load_all_csv_multithreaded(self, sample_ratio: float = 1.0) -> pd.DataFrame:
        """Loads all CSV files in the directory using multi-threading and resource monitoring."""
        print("\n" + "="*80)
        print(f"MICRO-TÂCHE: Chargement Multi-threadé des données depuis {self.data_dir}")
        print(f"STRATÉGIE: Un thread par fichier avec monitoring RAM (Limite 50%)")
        print("="*80)

        if self.data_dir.is_file():
            csv_files = [self.data_dir]
        else:
            csv_files = list(self.data_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {self.data_dir}")

        all_dfs = []
        start_time = time.time()

        # Adaptive thread count based on CPU
        max_workers = min(len(csv_files), (psutil.cpu_count() or 2))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self._load_single_file, f, sample_ratio): f for f in csv_files}

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    df_part = future.result()
                    if df_part is not None:
                        all_dfs.append(df_part)
                        print(f"RÉSULTAT: Fichier {file_path.name} chargé ({len(df_part)} lignes).")
                except Exception as exc:
                    print(f"ERREUR: {file_path.name} a généré une exception: {exc}")

        if not all_dfs:
            raise ValueError("Aucune donnée n'a pu être chargée.")

        print("\n" + "-"*40)
        print("MICRO-TÂCHE: Consolidation des données")
        self.df = pd.concat(all_dfs, ignore_index=True)
        print(f"RÉSULTAT: Dataset consolidé. Shape: {self.df.shape}. Temps total: {time.time() - start_time:.2f}s")

        # Save consolidated CSV
        consolidated_path = Path("output/consolidated_data.csv")
        consolidated_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(consolidated_path, index=False)
        print(f"RÉSULTAT: Fichier CSV consolidé sauvegardé dans {consolidated_path}")

        return self.df

    def _load_single_file(self, file_path: Path, sample_ratio: float) -> Optional[pd.DataFrame]:
        """Loads a single file while checking resource limits."""
        # Wait if RAM is too high
        while self.monitor.get_memory_info()['used_percent'] > self.monitor.max_memory_percent:
            time.sleep(0.5)

        try:
            df = pd.read_csv(file_path, low_memory=False)
            if sample_ratio < 1.0:
                df = df.sample(frac=sample_ratio, random_state=42)

            # Optimize memory immediately
            df = self._optimize_dtypes(df)
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduces memory usage by downcasting types."""
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        return df

    def profile_and_validate(self) -> dict:
        """Profiling, KS Validation and Class Distribution Plot."""
        if self.df is None:
            raise ValueError("Load data first.")

        total_rows = len(self.df)

        # Target identification
        if self.target_col not in self.df.columns:
            for c in ['label', 'Label', 'type', 'Attack']:
                if c in self.df.columns:
                    self.target_col = c
                    break

        self.df['is_ddos'] = (self.df[self.target_col].astype(str).str.lower() != 'normal').astype(int)
        counts = self.df['is_ddos'].value_counts()

        # Splits
        train_df, temp_df = train_test_split(self.df, test_size=0.4, stratify=self.df['is_ddos'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['is_ddos'], random_state=42)
        self.splits = {'train': train_df, 'val': val_df, 'test': test_df}

        # KS Validation
        self._validate_ks(train_df, test_df)

        # Plots
        self._plot_distributions(counts)

        return {
            'total_rows': total_rows,
            'split_counts': {k: len(v) for k, v in self.splits.items()}
        }

    def _validate_ks(self, train_df, test_df):
        print("\n" + "-"*40)
        print("MICRO-TÂCHE: Validation statistique KS Test")
        num_cols = train_df.select_dtypes(include=[np.number]).columns[:10]
        for col in num_cols:
            stat, p = ks_2samp(train_df[col].dropna(), test_df[col].dropna())
            # print(f"  Feature {col}: p-value={p:.4f}")

    def _plot_distributions(self, counts):
        plt.figure(figsize=(8, 6))
        sns.barplot(x=['Normal', 'DDoS'], y=[counts.get(0, 0), counts.get(1, 0)])
        plt.title("Class Distribution")
        plt.savefig(self.rr_dir / "phase1_distribution.png")
        plt.close()

    def get_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.splits is None:
            raise ValueError("Splits not generated.")
        return self.splits['train'], self.splits['val'], self.splits['test']
