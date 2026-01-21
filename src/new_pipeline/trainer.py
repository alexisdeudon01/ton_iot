import logging
import time
import joblib
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import recall_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.new_pipeline.config import config, ModelName
from src.core.results import TrainingResult

logger = logging.getLogger(__name__)

class LateFusionTrainer:
    """Phase 4: Training Séparé (LATE FUSION) avec Tuning et Calibration."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.p4 = config.phase4
        self.models_root = self.p4.models_root / dataset_name
        self.models_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized LateFusionTrainer for {dataset_name}")

    def train(self, train_path: Path, val_path: Path, model_name: ModelName, features: List[str]) -> TrainingResult:
        """Trains, tunes, and calibrates a model on a specific dataset."""
        logger.info(f"Starting training for {model_name} on {self.dataset_name}")
        start_time = time.time()
        
        df_train = pd.read_parquet(train_path)
        df_val = pd.read_parquet(val_path)
        
        X_train, y_train = df_train[features], df_train[config.phase0.label_col]
        X_val, y_val = df_val[features], df_val[config.phase0.label_col]
        
        # 1. Model Factory
        if model_name == "LR":
            base_model = LogisticRegression(max_iter=1000, random_state=self.p4.random_state)
        elif model_name == "DT":
            base_model = DecisionTreeClassifier(random_state=self.p4.random_state)
        elif model_name == "RF":
            base_model = RandomForestClassifier(random_state=self.p4.random_state)
        else:
            logger.warning(f"Model {model_name} not fully implemented, using RF fallback.")
            base_model = RandomForestClassifier(n_estimators=50, random_state=self.p4.random_state)

        # 2. Hyperparameter Tuning
        tuning_cfg = self.p4.tuning
        if tuning_cfg.mode != "none" and model_name in tuning_cfg.grids:
            logger.info(f"  - Tuning {model_name} using {tuning_cfg.mode} search...")
            grid = tuning_cfg.grids[model_name]
            
            if tuning_cfg.mode == "grid":
                search = GridSearchCV(base_model, grid, cv=tuning_cfg.cv_folds, scoring=tuning_cfg.primary_metric, n_jobs=-1)
            else:
                search = RandomizedSearchCV(base_model, grid, n_iter=tuning_cfg.n_iter_random, 
                                           cv=tuning_cfg.cv_folds, scoring=tuning_cfg.primary_metric, 
                                           random_state=self.p4.random_state, n_jobs=-1)
            
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            logger.info(f"    * Best params: {search.best_params_}")
        else:
            logger.info(f"  - Training {model_name} with default parameters...")
            best_model = base_model.fit(X_train, y_train)
        
        # 3. Calibration (OBLIGATOIRE pour Late Fusion propre)
        if self.p4.calibrate_probs:
            logger.info(f"  - Calibrating {model_name} probabilities (method: {self.p4.calibration_method})...")
            model = CalibratedClassifierCV(best_model, method=self.p4.calibration_method, cv="prefit")
            model.fit(X_val, y_val)
        else:
            model = best_model

        # 4. Persistence with Joblib
        model_path = self.models_root / f"{model_name.lower()}_model.joblib"
        joblib.dump(model, model_path)
        logger.info(f"  - Model saved to: {model_path}")
        
        # 5. Metrics on validation
        val_preds = model.predict(X_val)
        rec = recall_score(y_val, val_preds)
        f1 = f1_score(y_val, val_preds)
        
        elapsed = time.time() - start_time
        logger.info(f"Phase 4 completed for {model_name} in {elapsed:.2f}s. Recall: {rec:.4f}, F1: {f1:.4f}")
        
        return TrainingResult(
            model_name=model_name,
            success=True,
            training_time=elapsed,
            history={'recall': [rec], 'f1': [f1]},
            model_path=model_path
        )
