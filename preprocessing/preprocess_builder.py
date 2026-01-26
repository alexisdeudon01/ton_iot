from typing import Dict, List, Tuple

import pandas as pd

from preprocessing.imputer import DataImputer
from preprocessing.scaler import DataScaler
from preprocessing.encoder import DataEncoder


class PreprocessBuilder:
    """
    Constructeur de pipeline de prétraitement modulaire.
    """
    def __init__(self, config: Dict | None = None):
        cfg = config or {}
        self.use_cats = bool(cfg.get("use_cats", False))
        self.scaler = DataScaler()

    def execute(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Exécute la séquence complète de prétraitement.
        Retourne le DataFrame transformé et l'ordre exact des features (feature_order).
        """
        exclude = ["y", "source_file", "sample_id", "Label", "type"]

        numeric_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]
        cat_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_string_dtype(df[c])
        ]

        df = DataImputer.impute_median(df, numeric_cols)
        df = self.scaler.fit_transform(df, numeric_cols)

        if self.use_cats and cat_cols:
            df = DataEncoder.one_hot_encode(df, cat_cols)

        feature_order = [c for c in df.columns if c not in exclude]
        return df, feature_order
