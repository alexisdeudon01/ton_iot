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
        """Run Phase 3: Evaluate algorithms across 3 dimensions."""
        logger.info("Phase 3: 3D Evaluation (full implementation)")

        df_processed = self._load_and_prepare_dataset()
        X = df_processed.drop(
            ['label', 'dataset_source'] if 'dataset_source' in df_processed.columns else ['label'],
            axis=1,
            errors='ignore'
        )
        y = df_processed['label']

        preprocessing_result = self._run_preprocessing(X, y)
        X_processed = preprocessing_result['X_processed']
        y_processed = preprocessing_result['y_processed']
        feature_names = preprocessing_result['feature_names']

        models = self._build_models()
        if not models:
            raise ValueError("No algorithms available for evaluation. Check dependencies and config.")

        cv = StratifiedCrossValidator(
            n_splits=self.config.phase3_cv_folds,
            random_state=self.config.random_state
        )
        evaluator = Evaluation3D(feature_names=feature_names)

        logger.info(f"Evaluating {len(models)} algorithms with {self.config.phase3_cv_folds}-fold CV")
        all_results = []

        for model_name, model_template in tqdm(models.items(), desc="Evaluating algorithms"):
            logger.info("Evaluating %s...", model_name)
            fold_results = []

            for fold_idx, (train_idx, test_idx) in enumerate(
                tqdm(
                    cv.split(X_processed, y_processed),
                    desc=f"  {model_name} CV folds",
                    total=self.config.phase3_cv_folds,
                    leave=False
                )
            ):
                X_train_fold, X_test_fold = X_processed[train_idx], X_processed[test_idx]
                y_train_fold, y_test_fold = y_processed[train_idx], y_processed[test_idx]

                try:
                    results = evaluator.evaluate_model(
                        model_template,
                        f"{model_name}_fold{fold_idx + 1}",
                        X_train_fold,
                        y_train_fold,
                        X_test_fold,
                        y_test_fold,
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
        """Load, harmonize, and fuse datasets."""
        logger.info("Loading datasets for Phase 3...")
        df_ton = self.loader.load_ton_iot(
            sample_ratio=self.config.sample_ratio,
            random_state=self.config.random_state,
            incremental=False
        )

        try:
            df_cic = self.loader.load_cic_ddos2019(
                sample_ratio=self.config.sample_ratio,
                random_state=self.config.random_state,
                incremental=False,
                max_files_in_test=10 if self.config.test_mode else None
            )
        except FileNotFoundError:
            logger.warning("CIC-DDoS2019 not available. Using TON_IoT only.")
            df_cic = pd.DataFrame()

        if not df_cic.empty:
            try:
                df_cic_harm, df_ton_harm = self.harmonizer.harmonize_features(df_cic, df_ton)
                df_processed = self.harmonizer.early_fusion(df_cic_harm, df_ton_harm)
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

    def _run_preprocessing(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Run preprocessing using the best config if available."""
        best_config = self._load_best_config()
        feature_k = best_config.get('feature_selection_k') if best_config else None
        pipeline = PreprocessingPipeline(
            random_state=self.config.random_state,
            n_features=feature_k or 20
        )

        apply_encoding = best_config.get('apply_encoding', True) if best_config else True
        apply_feature_selection = best_config.get('apply_feature_selection', True) if best_config else True
        apply_scaling = best_config.get('apply_scaling', True) if best_config else True
        apply_resampling = best_config.get('apply_resampling', True) if best_config else True

        return pipeline.prepare_data(
            X,
            y,
            apply_encoding=apply_encoding,
            apply_feature_selection=apply_feature_selection,
            apply_scaling=apply_scaling,
            apply_resampling=apply_resampling,
            apply_splitting=False
        )

    def _load_best_config(self) -> Dict:
        """Load best preprocessing config from Phase 1 if available."""
        best_config_file = Path(self.config.output_dir) / 'phase1_config_search' / 'best_config.json'
        if not best_config_file.exists():
            logger.info("No Phase 1 best_config.json found; using default preprocessing settings.")
            return {}

        try:
            with open(best_config_file, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
            return data.get('config', {})
        except Exception as exc:
            logger.warning("Failed to load best_config.json: %s", exc)
            return {}

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
