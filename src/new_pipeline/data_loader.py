import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class RealDataLoader:
    """Phase 1: Chargement des Données Réelles et Profiling"""

    def __init__(self, file_path: str, target_col: str = 'label'):
        self.file_path = Path(file_path)
        self.target_col = target_col
        self.df = None

    def load_and_profile(self) -> dict:
        """Charge le dataset et génère un rapport de profiling."""
        logger.info(f"[PHASE 1] Chargement du dataset: {self.file_path}")

        if self.file_path.suffix == '.csv':
            self.df = pd.read_csv(self.file_path, low_memory=False)
        elif self.file_path.suffix == '.parquet':
            self.df = pd.read_parquet(self.file_path)
        else:
            raise ValueError("Format de fichier non supporté (CSV ou Parquet uniquement)")

        total_rows = len(self.df)

        # Nettoyage basique pour le profiling
        if self.target_col not in self.df.columns:
            # Essayer de trouver une colonne de label
            candidates = ['label', 'Label', 'type', 'Attack']
            for c in candidates:
                if c in self.df.columns:
                    self.target_col = c
                    break

        # Calcul de la proportion DDoS vs Normal
        # On suppose 0 = Normal, 1 = DDoS (ou 'normal' vs autres)
        if self.df[self.target_col].dtype == 'object':
            self.df['is_ddos'] = (self.df[self.target_col].str.lower() != 'normal').astype(int)
        else:
            self.df['is_ddos'] = (self.df[self.target_col] != 0).astype(int)

        counts = self.df['is_ddos'].value_counts()
        prop_normal = (counts.get(0, 0) / total_rows) * 100
        prop_ddos = (counts.get(1, 0) / total_rows) * 100

        # Splits
        train_df, test_df = train_test_split(self.df, test_size=0.2, stratify=self.df['is_ddos'], random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['is_ddos'], random_state=42)

        report = {
            'total_rows': total_rows,
            'prop_normal': prop_normal,
            'prop_ddos': prop_ddos,
            'split_counts': {
                'training': len(train_df),
                'validation': len(val_df),
                'testing': len(test_df)
            },
            'split_pct': {
                'training': (len(train_df) / total_rows) * 100,
                'validation': (len(val_df) / total_rows) * 100,
                'testing': (len(test_df) / total_rows) * 100
            }
        }

        self.splits = (train_df, val_df, test_df)

        logger.info(f"[REPORT PHASE 1] Total Rows: {total_rows}")
        logger.info(f"[REPORT PHASE 1] DDoS: {prop_ddos:.2f}%, Normal: {prop_normal:.2f}%")
        logger.info(f"[REPORT PHASE 1] Splits: Train={len(train_df)} ({report['split_pct']['training']:.1f}%), "
                    f"Val={len(val_df)} ({report['split_pct']['validation']:.1f}%), "
                    f"Test={len(test_df)} ({report['split_pct']['testing']:.1f}%)")

        return report

    def get_splits(self):
        return self.splits
