import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, Optional

logger = logging.getLogger(__name__)

class RealDataLoader:
    """Phase 1: Chargement des Données Réelles, Profiling et Validation Statistique"""

    def __init__(self, file_path: Path, target_col: str = 'label', rr_dir: Path = Path("rr")):
        self.file_path = file_path
        self.target_col = target_col
        self.rr_dir = rr_dir
        self.df: Optional[pd.DataFrame] = None
        self.splits: Optional[Dict[str, pd.DataFrame]] = None

    def load_and_profile(self) -> dict:
        """Charge le dataset, valide statistiquement et génère un rapport de profiling."""
        start_time = time.time()
        print("\n" + "="*80)
        print(f"MICRO-TÂCHE: Chargement du dataset depuis {self.file_path}")
        print("="*80)
        logger.info(f"[PHASE 1] Début du chargement. Input: {self.file_path}")

        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset non trouvé: {self.file_path}")

        if self.file_path.suffix == '.csv':
            self.df = pd.read_csv(self.file_path, low_memory=False)
        elif self.file_path.suffix == '.parquet':
            self.df = pd.read_parquet(self.file_path)
        else:
            raise ValueError("Format de fichier non supporté (CSV ou Parquet uniquement)")

        if self.df is None:
            raise ValueError("Échec du chargement du DataFrame.")

        total_rows = len(self.df)
        print(f"RÉSULTAT: Dataset chargé. Shape: {self.df.shape}. Temps: {time.time() - start_time:.2f}s")

        # Identification de la colonne cible
        print("\n" + "-"*40)
        print("MICRO-TÂCHE: Identification de la colonne cible et encodage binaire")
        if self.target_col not in self.df.columns:
            candidates = ['label', 'Label', 'type', 'Attack', 'class']
            for c in candidates:
                if c in self.df.columns:
                    self.target_col = c
                    break

        if self.df[self.target_col].dtype == 'object':
            self.df['is_ddos'] = (~self.df[self.target_col].str.lower().isin(['normal', 'benign'])).astype(int)
        else:
            self.df['is_ddos'] = (self.df[self.target_col] != 0).astype(int)

        counts = self.df['is_ddos'].value_counts()
        prop_normal = (counts.get(0, 0) / total_rows) * 100
        prop_ddos = (counts.get(1, 0) / total_rows) * 100
        print(f"RÉSULTAT: Cible '{self.target_col}' identifiée. DDoS: {prop_ddos:.2f}%, Normal: {prop_normal:.2f}%")

        # Sauvegarde du dataset harmonisé dans rr
        print("\n" + "-"*40)
        print(f"MICRO-TÂCHE: Sauvegarde du dataset harmonisé dans {self.rr_dir}")
        harmonized_path = self.rr_dir / "dataset_harmonized.parquet"
        self.df.to_parquet(harmonized_path, index=False)
        print(f"RÉSULTAT: Fichier sauvegardé: {harmonized_path}")

        # Splits Train (60%), Val (20%), Test (20%)
        print("\n" + "-"*40)
        print("MICRO-TÂCHE: Division du dataset (Train/Val/Test)")
        train_df, temp_df = train_test_split(self.df, test_size=0.4, stratify=self.df['is_ddos'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['is_ddos'], random_state=42)

        self.splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        print(f"RÉSULTAT: Splits générés. Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        # Validation Statistique (KS Test)
        self._validate_dataset_consistency(train_df, test_df)

        # Visualisation de la répartition
        self._plot_class_distribution(counts)

        report = {
            'total_rows': total_rows,
            'prop_normal': prop_normal,
            'prop_ddos': prop_ddos,
            'split_counts': {k: len(v) for k, v in self.splits.items()},
            'split_pct': {k: (len(v) / total_rows) * 100 for k, v in self.splits.items()}
        }

        return report

    def _validate_dataset_consistency(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Valide la cohérence des données entre Train et Test via KS Test."""
        print("\n" + "-"*40)
        print("MICRO-TÂCHE: Validation statistique KS Test (Train vs Test)")
        num_cols = train_df.select_dtypes(include=[np.number]).columns
        num_cols = [c for c in num_cols if c not in ['is_ddos', self.target_col]]

        inconsistent_features = []
        for col in num_cols[:20]:
            res = ks_2samp(train_df[col].dropna(), test_df[col].dropna())
            p_val = float(res.pvalue) # type: ignore
            if p_val < 0.05:
                inconsistent_features.append(col)

        if inconsistent_features:
            print(f"RÉSULTAT: {len(inconsistent_features)} features potentiellement incohérentes détectées.")
        else:
            print("RÉSULTAT: Distributions cohérentes entre Train et Test (p >= 0.05).")

    def _plot_class_distribution(self, counts: pd.Series):
        """Génère l'histogramme de répartition des classes."""
        print("\n" + "-"*40)
        print("MICRO-TÂCHE: Génération du graphique de répartition des classes")
        plt.figure(figsize=(8, 6))
        sns.barplot(x=['Normal', 'DDoS'], y=[counts.get(0, 0), counts.get(1, 0)], palette='viridis')
        plt.title("Répartition des Classes (Normal vs DDoS)")
        plt.ylabel("Nombre de lignes")
        plt.savefig(self.rr_dir / "phase1_class_distribution.png")
        plt.close()
        print(f"RÉSULTAT: Graphique sauvegardé dans {self.rr_dir}/phase1_class_distribution.png")

    def get_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.splits is None:
            raise ValueError("Les splits n'ont pas encore été générés.")
        return self.splits['train'], self.splits['val'], self.splits['test']
