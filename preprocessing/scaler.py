import polars as pl
from sklearn.preprocessing import RobustScaler
import pandas as pd
from typing import List

class DataScaler:
    """
    Mise à l'échelle robuste des caractéristiques numériques pour gérer les outliers fréquents en DDoS.
    """
    def __init__(self):
        self.scaler = RobustScaler()

    def fit_transform(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """Ajuste le scaler et transforme les colonnes spécifiées."""
        if not columns:
            return df
        
        # Conversion pandas pour compatibilité sklearn
        pdf = df.select(columns).to_pandas()
        scaled_data = self.scaler.fit_transform(pdf)
        
        scaled_df = pl.from_pandas(pd.DataFrame(scaled_data, columns=columns))
        
        # Fusion avec les colonnes non transformées
        other_cols = [c for c in df.columns if c not in columns]
        return df.select(other_cols).hstack(scaled_df)

    def transform(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """Transforme les colonnes en utilisant un scaler déjà ajusté."""
        if not columns:
            return df
            
        pdf = df.select(columns).to_pandas()
        scaled_data = self.scaler.transform(pdf)
        scaled_df = pl.from_pandas(pd.DataFrame(scaled_data, columns=columns))
        
        other_cols = [c for c in df.columns if c not in columns]
        return df.select(other_cols).hstack(scaled_df)
