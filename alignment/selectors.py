from typing import List

import pandas as pd

class FeatureSelector:
    """
    Sélectionne et projette les DataFrames sur les colonnes alignées.
    """
    @staticmethod
    def project_to_common(df: pd.DataFrame, common_features: List[str]) -> pd.DataFrame:
        """
        Garantit que le DataFrame contient exactement les features communes + la cible y.
        """
        required = common_features + ["y", "source_file"]
        available = [c for c in required if c in df.columns]
        return df.loc[:, available].copy()
