from typing import List

import pandas as pd

class DataImputer:
    """
    Imputation des valeurs manquantes par la médiane pour les colonnes numériques.
    """
    @staticmethod
    def impute_median(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        for col in columns:
            if col in df.columns:
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df[col] = df[col].fillna(median_val)
        return df
