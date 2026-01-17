#!/usr/bin/env python3
"""
Phase 3: 3D Evaluation
Évaluation des 5 algorithmes (LR/DT/RF/CNN/TabNet) en cross-validation
"""
import logging
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core import DatasetLoader, DataHarmonizer, PreprocessingPipeline
from src.core.preprocessing_pipeline import StratifiedCrossValidator
from src.core.model_utils import fresh_model
from src.evaluation_3d import Evaluation3D

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from src.models_cnn import CNNTabularClassifier, TORCH_AVAILABLE as CNN_AVAILABLE
except ImportError:
    CNNTabularClassifier = None
    CNN_AVAILABLE = False

try:
    from src.models_tabnet import TabNetClassifierWrapper
    TABNET_AVAILABLE = True
except ImportError:
    TabNetClassifierWrapper = None
    TABNET_AVAILABLE = False

logger = logging.getLogger(__name__)


class Phase3Evaluation:
    """Phase 3: Évaluation 3D"""
    
    def __init__(self, config):
        self.config = config
        self.results_dir = Path(config.output_dir) / 'phase3_evaluation'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.loader = DatasetLoader()
        self.harmonizer = DataHarmonizer()
    
    def run(self) -> Dict:
        """Run Phase 3: Evaluate algorithms across 3 dimensions with model-aware preprocessing per fold."""
        logger.info("Phase 3: 3D Evaluation (model-aware preprocessing per fold)")

        # Load dataset (from Phase2 if available, otherwise fallback)
        df_processed = self._load_and_prepare_dataset()
        X = df_processed.drop(
            ['label', 'dataset_source'] if 'dataset_source' in df_processed.columns else ['label'],
            axis=1,
            errors='ignore'
        )
        y = df_processed['label']

        # Get feature names (will be updated after preprocessing in each fold)
        feature_names = X.columns.tolist()

        models = self._build_models()
        if not models:
            raise ValueError("No algorithms available for evaluation. Check dependencies and config.")

        cv = StratifiedCrossValidator(
            n_splits=self.config.phase3_cv_folds,
            random_state=self.config.random_state
        )

        # Initialize evaluator (feature_names will be updated per fold)
        evaluator = Evaluation3D(feature_names=feature_names)

        logger.info(f"Evaluating {len(models)} algorithms with {self.config.phase3_cv_folds}-fold CV")
        logger.info("NOTE: Preprocessing (scaling, feature selection, SMOTE) applied per fold to prevent data leakage")
        all_results = []

        for model_name, model_template in tqdm(models.items(), desc="Evaluating algorithms"):
            logger.info("Evaluating %s...", model_name)
            
            # Get preprocessing profile for this model
            profile = self._get_preprocessing_profile(model_name, X.shape[1])
            
            fold_results = []

            for fold_idx, (train_idx, test_idx) in enumerate(
                tqdm(
                    cv.split(X.values, y.values),
                    desc=f"  {model_name} CV folds",
                    total=self.config.phase3_cv_folds,
                    leave=False
                )
            ):
                # Split data (before preprocessing)
                X_train_fold = X.iloc[train_idx].copy()
                X_test_fold = X.iloc[test_idx].copy()
                y_train_fold = y.iloc[train_idx].copy()
                y_test_fold = y.iloc[test_idx].copy()

                try:
                    # Apply model-aware preprocessing on TRAIN only
                    X_train_prep, y_train_prep, pipeline = self._apply_preprocessing_per_fold(
                        X_train_fold, y_train_fold, profile
                    )
                    
                    # Transform TEST with scaler/selector fitté on TRAIN
                    X_test_prep = pipeline.transform_test(X_test_fold.values)
                    y_test_prep = y_test_fold.values
                    
                    # Get fresh model instance for this fold
                    model_fold = fresh_model(model_template)
                    
                    # Apply class_weight if Tree profile
                    if profile.get('use_class_weight', False):
                        class_weight = profile.get('class_weight', 'balanced')
                        if hasattr(model_fold, 'set_params'):
                            model_fold.set_params(class_weight=class_weight)
                    
                    # Update evaluator with current feature names
                    if hasattr(pipeline, 'selected_features') and pipeline.selected_features:
                        evaluator.feature_names = pipeline.selected_features
                    else:
                        # If no feature selection, use original feature names
                        evaluator.feature_names = feature_names
                    
                    # Evaluate model (X_train_prep and X_test_prep are numpy arrays)
                    results = evaluator.evaluate_model(
                        model_fold,
                        f"{model_name}_fold{fold_idx + 1}",
                        X_train_prep,
                        y_train_prep,
                        X_test_prep,
                        y_test_prep,
                        compute_explainability=False
                    )
                    results['fold'] = fold_idx + 1
                    fold_results.append(results)
                except Exception as exc:
                    logger.warning("Fold %s failed for %s: %s", fold_idx + 1, model_name, exc)

            if fold_results:
                avg_results = {
                    'model_name': model_name,
                    'f1_score': float(np.mean([r['f1_score'] for r in fold_results])),
                    'accuracy': float(np.mean([r['accuracy'] for r in fold_results])),
                    'precision': float(np.mean([r['precision'] for r in fold_results])),
                    'recall': float(np.mean([r['recall'] for r in fold_results])),
                    'training_time_seconds': float(np.mean([r['training_time_seconds'] for r in fold_results])),
                    'memory_used_mb': float(np.mean([r['memory_used_mb'] for r in fold_results])),
                    'explainability_score': float(np.mean([r['explainability_score'] for r in fold_results])),
                    'n_successful_folds': len(fold_results)
                }
                all_results.append(avg_results)
                logger.info(
                    "%s - F1: %.4f, Time: %.2fs, Explainability: %.4f (%s/%s folds)",
                    model_name,
                    avg_results['f1_score'],
                    avg_results['training_time_seconds'],
                    avg_results['explainability_score'],
                    len(fold_results),
                    self.config.phase3_cv_folds
                )
            else:
                logger.error("All folds failed for %s; skipping aggregation.", model_name)

        results_df = self._save_results(all_results)
        self._generate_outputs(results_df, evaluator)

        return {
            'evaluation_results': results_df,
            'dimension_scores': evaluator.get_dimension_scores().to_dict(orient='records')
        }

    def _load_and_prepare_dataset(self) -> pd.DataFrame:
        """Load dataset from Phase2 if available, otherwise load and harmonize, or use synthetic data."""
        # Check if synthetic mode is enabled
        if getattr(self.config, 'synthetic_mode', False):
            logger.info("Using synthetic dataset for Phase 3 evaluation...")
            return self._generate_synthetic_dataset()
        
        # Try to load from Phase2 first
        phase2_data_file = Path(self.config.output_dir) / 'phase2_apply_best_config' / 'best_preprocessed.parquet'
        phase2_data_file_csv = Path(self.config.output_dir) / 'phase2_apply_best_config' / 'best_preprocessed.csv.gz'
        
        if phase2_data_file.exists():
            logger.info("Loading preprocessed dataset from Phase2 (parquet)...")
            try:
                df_processed = pd.read_parquet(phase2_data_file, engine='pyarrow')
                logger.info(f"Loaded Phase2 dataset: {df_processed.shape}")
                return df_processed
            except Exception as exc:
                logger.warning(f"Failed to load Phase2 parquet ({exc}), trying csv.gz...")
        
        if phase2_data_file_csv.exists():
            logger.info("Loading preprocessed dataset from Phase2 (csv.gz)...")
            try:
                df_processed = pd.read_csv(phase2_data_file_csv, compression='gzip')
                logger.info(f"Loaded Phase2 dataset: {df_processed.shape}")
                return df_processed
            except Exception as exc:
                logger.warning(f"Failed to load Phase2 csv.gz ({exc}), falling back to direct loading...")
        
        # Fallback: load and harmonize directly
        logger.info("Phase2 dataset not found, loading and harmonizing datasets directly...")
        df_ton = self.loader.load_ton_iot(
            sample_ratio=self.config.sample_ratio,
            random_state=self.config.random_state,
            incremental=False
        )

        try:
            max_files = getattr(self.config, 'cic_max_files', None)
            if self.config.test_mode and max_files is None:
                max_files = 3
            
            df_cic = self.loader.load_cic_ddos2019(
                sample_ratio=self.config.sample_ratio,
                random_state=self.config.random_state,
                incremental=False,
                max_files_in_test=max_files
            )
        except FileNotFoundError:
            logger.warning("CIC-DDoS2019 not available. Using TON_IoT only.")
            df_cic = pd.DataFrame()

        if not df_cic.empty:
            try:
                df_cic_harm, df_ton_harm = self.harmonizer.harmonize_features(df_cic, df_ton)
                df_processed, _ = self.harmonizer.early_fusion(df_cic_harm, df_ton_harm)
            except Exception as exc:
                logger.warning("Harmonization failed, falling back to TON_IoT only: %s", exc)
                df_processed = df_ton
        else:
            df_processed = df_ton

        if 'label' not in df_processed.columns:
            for candidate in ['Label', 'label', 'Attack', 'Class']:
                if candidate in df_processed.columns:
                    df_processed = df_processed.rename(columns={candidate: 'label'})
                    break

        if 'label' not in df_processed.columns:
            raise ValueError("Label column not found in dataset for Phase 3 evaluation.")

        return df_processed
    
    def _generate_synthetic_dataset(self) -> pd.DataFrame:
        """Generate synthetic dataset using sklearn.datasets.make_classification."""
        from sklearn.datasets import make_classification
        
        logger.info("Generating synthetic dataset...")
        
        # Generate synthetic dataset
        n_samples = 5000 if self.config.test_mode else 10000
        n_features = 20
        n_informative = 15
        n_redundant = 5
        n_classes = 2
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=n_classes,
            random_state=self.config.random_state,
            class_sep=1.0  # Good separation between classes
        )
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['label'] = y
        
        logger.info(f"Generated synthetic dataset: {df.shape[0]} rows, {df.shape[1]-1} features")
        logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
        return df
    
    def _get_preprocessing_profile(self, model_name: str, n_features: int) -> Dict:
        """Get preprocessing profile for a model with dynamic feature selection."""
        model_name_lower = model_name.lower()
        
        # Determine profile type
        if 'logistic' in model_name_lower or 'lr' in model_name_lower:
            profile = self.config.preprocessing_profiles.get('lr_profile', {}).copy()
        elif 'tree' in model_name_lower or 'forest' in model_name_lower or 'decision' in model_name_lower:
            profile = self.config.preprocessing_profiles.get('tree_profile', {}).copy()
        elif 'cnn' in model_name_lower or 'tabnet' in model_name_lower:
            profile = self.config.preprocessing_profiles.get('nn_profile', {}).copy()
        else:
            # Default to LR profile
            profile = self.config.preprocessing_profiles.get('lr_profile', {}).copy()
        
        # Calculate feature_selection_k dynamically if needed
        if profile.get('feature_selection_k_dynamic', False):
            profile['feature_selection_k'] = min(60, max(10, int(0.3 * n_features)))
            profile.pop('feature_selection_k_dynamic', None)
        
        return profile
    
    def _apply_preprocessing_per_fold(self, X_train: pd.DataFrame, y_train: pd.Series, profile: Dict) -> tuple:
        """Apply model-aware preprocessing on training data only."""
        # Calculate feature_selection_k if dynamic
        feature_k = profile.get('feature_selection_k', 20)
        if profile.get('feature_selection_k_dynamic', False):
            feature_k = min(60, max(10, int(0.3 * X_train.shape[1])))
        
        # Create new pipeline for this fold
        pipeline = PreprocessingPipeline(
            random_state=self.config.random_state,
            n_features=feature_k
        )
        
        # Apply preprocessing according to profile
        result = pipeline.prepare_data(
            X_train,
            y_train,
            apply_encoding=False,  # Already encoded in Phase2
            apply_feature_selection=profile.get('apply_feature_selection', True),
            apply_scaling=profile.get('apply_scaling', True),
            apply_resampling=profile.get('apply_resampling', True),
            apply_splitting=False
        )
        
        X_train_prep = result['X_processed']
        y_train_prep = result['y_processed']
        
        return X_train_prep, y_train_prep, pipeline

    def _build_models(self) -> Dict[str, object]:
        """Build model dictionary based on config and availability."""
        requested = {name.lower().replace("-", "_") for name in self.config.phase3_algorithms}
        models = {}

        def enabled(key: str) -> bool:
            return key.lower().replace("-", "_") in requested

        if enabled('logistic_regression'):
            models['Logistic Regression'] = LogisticRegression(
                max_iter=1000,
                random_state=self.config.random_state
            )
        if enabled('decision_tree'):
            models['Decision Tree'] = DecisionTreeClassifier(random_state=self.config.random_state)
        if enabled('random_forest'):
            models['Random Forest'] = RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_state
            )

        if enabled('cnn'):
            if CNN_AVAILABLE and CNNTabularClassifier is not None:
                try:
                    models['CNN'] = CNNTabularClassifier(
                        epochs=20,
                        batch_size=64,
                        random_state=self.config.random_state
                    )
                except (ImportError, AttributeError) as exc:
                    logger.warning("CNN not available: %s", exc)
            else:
                logger.warning("CNN skipped (torch not available).")

        if enabled('tabnet'):
            if TABNET_AVAILABLE and TabNetClassifierWrapper is not None:
                try:
                    models['TabNet'] = TabNetClassifierWrapper(
                        max_epochs=50,
                        batch_size=1024,
                        seed=self.config.random_state,
                        verbose=0
                    )
                except (ImportError, AttributeError) as exc:
                    logger.warning("TabNet not available: %s", exc)
            else:
                logger.warning("TabNet skipped (pytorch-tabnet not available).")

        return models

    def _save_results(self, all_results: list) -> pd.DataFrame:
        """Save evaluation results and return DataFrame."""
        if not all_results:
            logger.error("No evaluation results collected. Writing placeholder row.")
            all_results = [{
                'model_name': 'NO_MODELS_EVALUATED',
                'f1_score': np.nan,
                'accuracy': np.nan,
                'precision': np.nan,
                'recall': np.nan,
                'training_time_seconds': np.nan,
                'memory_used_mb': np.nan,
                'explainability_score': np.nan,
                'n_successful_folds': 0
            }]

        results_df = pd.DataFrame(all_results)
        output_file = self.results_dir / 'evaluation_results.csv'
        results_df.to_csv(output_file, index=False)
        logger.info("Saved evaluation results to %s", output_file)

        return results_df

    def _generate_outputs(self, results_df: pd.DataFrame, evaluator: Evaluation3D) -> None:
        """Generate dimension scores, reports, and visualizations."""
        try:
            dimension_scores = evaluator.get_dimension_scores()
            dimension_scores_file = self.results_dir / 'dimension_scores.csv'
            dimension_scores.to_csv(dimension_scores_file, index=False)
            logger.info("Saved dimension scores to %s", dimension_scores_file)
        except Exception as exc:
            logger.warning("Error generating dimension scores: %s", exc)

        try:
            for _, row in results_df.iterrows():
                evaluator.generate_algorithm_report(
                    row['model_name'],
                    row.to_dict(),
                    self.results_dir
                )
        except Exception as exc:
            logger.warning("Error generating algorithm reports: %s", exc)

        try:
            evaluator.generate_dimension_visualizations(self.results_dir)
        except Exception as exc:
            logger.warning("Error generating visualizations: %s", exc)
