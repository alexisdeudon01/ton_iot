import logging
import json
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
    """Phase 2: Preprocessing (Par Dataset) avec RobustScaler et Imputation Médiane."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.p2 = config.phase2
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy=self.p2.impute_numeric_strategy)
        logger.info(f"Initialized Preprocessor for {dataset_name} (Scaler: Robust, Imputer: Median)")

    def process(self, ddf: dd.DataFrame, features: List[str]) -> Dict[str, Path]:
        """Impute, Scale and Split with verbose logging."""
        logger.info(f"Starting preprocessing for dataset: {self.dataset_name}")
        logger.info(f"Target features count: {len(features)}")
        
        # Loading data into memory for preprocessing (as per requirements for RobustScaler)
        start_load = time.time()
        df = ddf[features + [config.phase0.label_col]].compute()
        logger.info(f"Data loaded in {time.time() - start_load:.2f}s. Shape: {df.shape}")
        
        X = df[features]
        y = df[config.phase0.label_col]
        
        # Imputation
        logger.info("Applying median imputation...")
        X_imputed = self.imputer.fit_transform(X)
        
        # Scaling
        logger.info("Applying RobustScaler (OBLIGATOIRE)...")
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Split
        logger.info(f"Splitting data (Test size: {self.p2.test_size}, Stratified: {self.p2.stratify})")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_scaled, y, test_size=self.p2.test_size, stratify=y, random_state=self.p2.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=self.p2.random_state
        )
        
        # Save splits
        dataset_root = self.p2.splits_root / self.dataset_name
        dataset_root.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        for name, data_X, data_y in [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]:
            df_split = pd.DataFrame(data_X, columns=features)
            df_split[config.phase0.label_col] = data_y.values
            path = dataset_root / f"{name}.parquet"
            df_split.to_parquet(path, index=False)
            paths[name] = path
            logger.info(f"  - Saved {name} split: {path} ({len(df_split)} rows)")
            
        logger.info(f"Preprocessing for {self.dataset_name} completed successfully.")
        return paths

import time # Ensure time is available
