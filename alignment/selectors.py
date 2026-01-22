import polars as pl
from typing import List

class FeatureSelector:
    """
    Sélectionne et projette les DataFrames sur les colonnes alignées.
    """
    @staticmethod
    def project_to_common(df: pl.DataFrame, common_features: List[str]) -> pl.DataFrame:
        """
        Garantit que le DataFrame contient exactement les features communes + la cible y.
        """
        required = common_features + ["y", "source_file"]
        # On ne garde que ce qui est disponible
        available = [c for c in required if c in df.columns]
        return df.select(available)
