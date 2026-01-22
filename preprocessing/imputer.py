import polars as pl
from typing import List

class DataImputer:
    """
    Imputation des valeurs manquantes par la médiane pour les colonnes numériques.
    """
    @staticmethod
    def impute_median(df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        for col in columns:
            if col in df.columns:
                median_val = df[col].median()
                if median_val is not None:
                    df = df.with_columns(pl.col(col).fill_null(median_val))
                else:
                    # Fallback si toute la colonne est nulle
                    df = df.with_columns(pl.col(col).fill_null(0.0))
        return df
