from typing import List

import pandas as pd

class DataEncoder:
    """
    Encodage des variables catégorielles pour les modèles ML.
    """
    @staticmethod
    def one_hot_encode(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Applique un encodage One-Hot (dummies) sur les colonnes spécifiées.
        """
        if not columns:
            return df
        
        # Vérification de l'existence des colonnes
        existing_cols = [c for c in columns if c in df.columns]
        if not existing_cols:
            return df
            
        return pd.get_dummies(df, columns=existing_cols, drop_first=False)
