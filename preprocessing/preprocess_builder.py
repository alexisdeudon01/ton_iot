import polars as pl
from typing import List, Tuple
from preprocessing.imputer import DataImputer
from preprocessing.scaler import DataScaler
from preprocessing.encoder import DataEncoder
from config.schema import PreprocessConfig

class PreprocessBuilder:
    """
    Constructeur de pipeline de prétraitement modulaire.
    """
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.scaler = DataScaler()

    def execute(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, List[str]]:
        """
        Exécute la séquence complète de prétraitement.
        Retourne le DataFrame transformé et l'ordre exact des features (feature_order).
        """
        # Identification des types de colonnes
        exclude = ["y", "source_file", "sample_id", "Label", "type"]
        numeric_cols = [c for c, t in zip(df.columns, df.dtypes) if t.is_numeric() and c not in exclude]
        cat_cols = [c for c, t in zip(df.columns, df.dtypes) if t == pl.String and c not in exclude]

        # 1. Imputation (Médiane)
        df = DataImputer.impute_median(df, numeric_cols)

        # 2. Scaling (Robust)
        df = self.scaler.fit_transform(df, numeric_cols)

        # 3. Encoding (OneHot si activé)
        if self.config.use_cats and cat_cols:
            df = DataEncoder.one_hot_encode(df, cat_cols)

        # Définition de l'ordre final des features (important pour la reproductibilité)
        feature_order = [c for c in df.columns if c not in exclude]
        
        return df, feature_order
