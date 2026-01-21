import logging
import json
import time
import joblib
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import dask.dataframe as dd
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.new_pipeline.config import config

logger = logging.getLogger(__name__)

class DatasetPreprocessor:
    """
    Phase 2: Preprocessing (Par Dataset).
    CORRECTION: Split avant Fit pour éviter la fuite de données (Data Leakage).
    Gère les valeurs Inf/NaN résiduelles de manière agressive.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.p2 = config.phase2
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy=self.p2.impute_numeric_strategy)
        self.artifacts_dir = config.paths.artifacts_root / "preprocessing" / dataset_name
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized Preprocessor for {dataset_name}")

    def process(self, ddf: dd.DataFrame, features: List[str]) -> Dict[str, Path]:
        """Exécute le pipeline de preprocessing : Split -> Fit (Train) -> Transform (All)."""
        logger.info(f"Starting preprocessing for {self.dataset_name} (Leakage Protected)")
        
        if not features:
            raise ValueError(f"No features provided for preprocessing {self.dataset_name}")

        # 1. Chargement et nettoyage final
        df = ddf[features + [config.phase0.label_col]].compute()
        # Sécurité maximale contre les Inf et NaN avant sklearn
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0.0, inplace=True)
        
        X = df[features]
        y = df[config.phase0.label_col]

        # 2. Split initial (Train / Temp)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=self.p2.test_size + self.p2.val_size, 
            stratify=y, 
            random_state=self.p2.random_state
        )

        # 3. Split Temp (Val / Test)
        relative_val_size = self.p2.val_size / (self.p2.test_size + self.p2.val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=1.0 - relative_val_size, 
            stratify=y_temp, 
            random_state=self.p2.random_state
        )

        logger.info(f"Splits created: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        # 4. Fit sur Train uniquement
        logger.info("Fitting imputer and scaler on TRAIN set only...")
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)

        # 5. Transform sur Val et Test
        X_val_imputed = self.imputer.transform(X_val)
        X_val_scaled = self.scaler.transform(X_val_imputed)
        
        X_test_imputed = self.imputer.transform(X_test)
        X_test_scaled = self.scaler.transform(X_test_imputed)

        # 6. Sauvegarde des artefacts
        joblib.dump(self.imputer, self.artifacts_dir / "imputer.joblib")
        joblib.dump(self.scaler, self.artifacts_dir / "scaler.joblib")

        # 7. Sauvegarde des splits
        dataset_root = self.p2.splits_root / self.dataset_name
        dataset_root.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        split_data = [('train', X_train_scaled, y_train), ('val', X_val_scaled, y_val), ('test', X_test_scaled, y_test)]

        for name, data_X, data_y in split_data:
            df_split = pd.DataFrame(data_X, columns=features)
            df_split[config.phase0.label_col] = data_y.values
            path = dataset_root / f"{name}.parquet"
            df_split.to_parquet(path, index=False)
            paths[name] = path
            
        logger.info(f"Preprocessing for {self.dataset_name} completed.")
        return paths
