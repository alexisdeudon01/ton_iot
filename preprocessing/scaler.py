from typing import List

import pandas as pd
from sklearn.preprocessing import RobustScaler

class DataScaler:
    """
    Mise à l'échelle robuste des caractéristiques numériques pour gérer les outliers fréquents en DDoS.
    """
    def __init__(self):
        self.scaler = RobustScaler()

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Ajuste le scaler et transforme les colonnes spécifiées."""
        if not columns:
            return df
        
        scaled_data = self.scaler.fit_transform(df[columns])
        df_out = df.copy()
        df_out[columns] = scaled_data
        return df_out

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Transforme les colonnes en utilisant un scaler déjà ajusté."""
        if not columns:
            return df
            
        scaled_data = self.scaler.transform(df[columns])
        df_out = df.copy()
        df_out[columns] = scaled_data
        return df_out
