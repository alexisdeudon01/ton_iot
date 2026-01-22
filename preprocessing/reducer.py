import polars as pl
from sklearn.decomposition import PCA
import pandas as pd
from typing import List

class DimReducer:
    """
    Réduction de dimension via PCA pour optimiser le clustering HDBSCAN.
    """
    def __init__(self, variance: float = 0.95):
        self.variance = variance
        self.pca = PCA(n_components=variance)

    def fit_transform(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """
        Ajuste la PCA et réduit les dimensions des colonnes numériques.
        """
        if not columns or len(columns) < 2:
            return df
            
        pdf = df.select(columns).to_pandas()
        reduced_data = self.pca.fit_transform(pdf)
        
        new_cols = [f"pca_{i}" for i in range(reduced_data.shape[1])]
        reduced_df = pl.from_pandas(pd.DataFrame(reduced_data, columns=new_cols))
        
        # Conservation des colonnes non transformées
        other_cols = [c for c in df.columns if c not in columns]
        return df.select(other_cols).hstack(reduced_df)

    def transform(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """
        Applique une PCA déjà ajustée.
        """
        if not columns:
            return df
            
        pdf = df.select(columns).to_pandas()
        reduced_data = self.pca.transform(pdf)
        
        new_cols = [f"pca_{i}" for i in range(reduced_data.shape[1])]
        reduced_df = pl.from_pandas(pd.DataFrame(reduced_data, columns=new_cols))
        
        other_cols = [c for c in df.columns if c not in columns]
        return df.select(other_cols).hstack(reduced_df)
