import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class RealDataLoader:
    """Phase 1: Chargement des Données Réelles, Profiling et Validation Statistique"""

    def __init__(self, file_path: str, target_col: str = 'label', rr_dir: Path = Path("rr")):
        self.file_path = Path(file_path)
        self.target_col = target_col
        self.rr_dir = rr_dir
        self.df = None
        self.splits = None

    def load_and_profile(self) -> dict:
        """Charge le dataset, valide statistiquement et génère un rapport de profiling."""
        logger.info(f"[PHASE 1] Chargement du dataset: {self.file_path}")

        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset non trouvé: {self.file_path}")

        if self.file_path.suffix == '.csv':
            self.df = pd.read_csv(self.file_path, low_memory=False)
        elif self.file_path.suffix == '.parquet':
            self.df = pd.read_parquet(self.file_path)
        else:
            raise ValueError("Format de fichier non supporté (CSV ou Parquet uniquement)")

        total_rows = len(self.df)

        # Identification de la colonne cible
        if self.target_col not in self.df.columns:
            candidates = ['label', 'Label', 'type', 'Attack', 'class']
            for c in candidates:
                if c in self.df.columns:
                    self.target_col = c
                    break

        # Encodage binaire DDoS (1) vs Normal (0)
        if self.df[self.target_col].dtype == 'object':
            # On suppose que 'normal' ou 'benign' est la classe 0
            self.df['is_ddos'] = (~self.df[self.target_col].str.lower().isin(['normal', 'benign'])).astype(int)
        else:
            self.df['is_ddos'] = (self.df[self.target_col] != 0).astype(int)

        counts = self.df['is_ddos'].value_counts()
        prop_normal = (counts.get(0, 0) / total_rows) * 100
        prop_ddos = (counts.get(1, 0) / total_rows) * 100

        # Splits Train (60%), Val (20%), Test (20%)
        train_df, temp_df = train_test_split(self.df, test_size=0.4, stratify=self.df['is_ddos'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['is_ddos'], random_state=42)

        self.splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

        # Validation Statistique (KS Test) entre Train et Test sur les colonnes numériques
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

        logger.info(f"[REPORT PHASE 1] Total Rows: {total_rows}")
        logger.info(f"[REPORT PHASE 1] DDoS: {prop_ddos:.2f}%, Normal: {prop_normal:.2f}%")
        logger.info(f"[REPORT PHASE 1] Splits: Train={len(train_df)} ({report['split_pct']['train']:.1f}%), "
                    f"Val={len(val_df)} ({report['split_pct']['val']:.1f}%), "
                    f"Test={len(test_df)} ({report['split_pct']['test']:.1f}%)")

        return report

    def _validate_dataset_consistency(self, train_df, test_df):
        """Valide la cohérence des données entre Train et Test via KS Test."""
        logger.info("[PHASE 1] Validation statistique (KS Test) Train vs Test...")
        num_cols = train_df.select_dtypes(include=[np.number]).columns
        num_cols = [c for c in num_cols if c not in ['is_ddos', self.target_col]]

        inconsistent_features = []
        for col in num_cols[:20]: # Limiter aux 20 premières pour la rapidité
            stat, p_val = ks_2samp(train_df[col].dropna(), test_df[col].dropna())
            if p_val < 0.05:
                inconsistent_features.append(col)

        if inconsistent_features:
            logger.warning(f"[WARNING] Features potentiellement incohérentes (p < 0.05): {len(inconsistent_features)}")
        else:
            logger.info("[SUCCESS] Distributions cohérentes entre Train et Test.")

    def _plot_class_distribution(self, counts):
        """Génère l'histogramme de répartition des classes."""
        plt.figure(figsize=(8, 6))
        sns.barplot(x=['Normal', 'DDoS'], y=[counts.get(0, 0), counts.get(1, 0)], palette='viridis')
        plt.title("Répartition des Classes (Normal vs DDoS)")
        plt.ylabel("Nombre de lignes")
        plt.savefig(self.rr_dir / "phase1_class_distribution.png")
        plt.close()

    def get_splits(self):
        return self.splits['train'], self.splits['val'], self.splits['test']
