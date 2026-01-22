import polars as pl
from typing import List

class DataEncoder:
    """
    Encodage des variables catégorielles pour les modèles ML.
    """
    @staticmethod
    def one_hot_encode(df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """
        Applique un encodage One-Hot (dummies) sur les colonnes spécifiées.
        """
        if not columns:
            return df
        
        # Vérification de l'existence des colonnes
        existing_cols = [c for c in columns if c in df.columns]
        if not existing_cols:
            return df
            
        return df.to_dummies(existing_cols)
