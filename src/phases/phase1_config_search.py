py#!/usr/bin/env python3
"""
Phase 1: Preprocessing Configuration Search
Génère et évalue 108 configurations preprocessing pour trouver la meilleure
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from ..config import PipelineConfig, generate_108_configs

logger = logging.getLogger(__name__)

# Import modules (temporairement depuis src/, à déplacer dans core/ plus tard)
import sys
from pathlib import Path as PathLib
sys.path.insert(0, str(PathLib(__file__).parent.parent))
from dataset_loader import DatasetLoader
from data_harmonization import DataHarmonizer
from preprocessing_pipeline import PreprocessingPipeline


class Phase1ConfigSearch:
    """Phase 1: Recherche de la meilleure configuration preprocessing"""
    
    def __init__(self, config: PipelineConfig, loader: Optional[DatasetLoader] = None, 
                 harmonizer: Optional[DataHarmonizer] = None):
        """
        Initialize Phase 1
        
        Args:
            config: Pipeline configuration
            loader: Dataset loader (optional, will create if None)
            harmonizer: Data harmonizer (optional, will create if None)
        """
        self.config = config
        self.results_dir = Path(config.output_dir) / 'phase1_config_search'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.loader = loader or DatasetLoader()
        self.harmonizer = harmonizer or DataHarmonizer()
        
        self.configs = generate_108_configs()
        self.results = []
        
        logger.info(f"Phase 1 initialized: {len(self.configs)} configurations to evaluate")
    
    def run(self) -> Dict:
        """
        Run Phase 1: évaluer toutes les configurations
        
        Returns:
            Dict with best_config and evaluation_results
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: Preprocessing Configuration Search")
        logger.info("=" * 60)
        
        # Load and harmonize datasets
        logger.info("\n1.1 Loading and harmonizing datasets...")
        df_cic, df_ton = self._load_and_harmonize()
        
        # Early fusion
        logger.info("\n1.2 Early fusion...")
        df_fused = self._early_fusion(df_cic, df_ton)
        
        # Prepare features/labels
        X = df_fused.drop('label', axis=1)
        y = df_fused['label']
        
        logger.info(f"  Fused dataset: {X.shape}")
        logger.info(f"  Class distribution: {y.value_counts().to_dict()}")
        
        # Evaluate each configuration
        logger.info(f"\n1.3 Evaluating {len(self.configs)} configurations...")
        
        best_score = -np.inf
        best_config = None
        best_config_id = None
        
        for config_id, preproc_config in enumerate(tqdm(self.configs, desc="Configs")):
            try:
                score = self._evaluate_config(X, y, preproc_config, config_id)
                
                result = {
                    'config_id': config_id,
                    'config': preproc_config,
                    'score': score
                }
                self.results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_config = preproc_config
                    best_config_id = config_id
                    
            except Exception as e:
                logger.warning(f"  Config {config_id} failed: {e}")
                self.results.append({
                    'config_id': config_id,
                    'config': preproc_config,
                    'score': -np.inf,
                    'error': str(e)
                })
        
        # Save results
        self._save_results(best_config, best_config_id, best_score)
        
        logger.info(f"\n✅ Phase 1 complete: Best config ID {best_config_id} with score {best_score:.4f}")
        
        return {
            'best_config': best_config,
            'best_config_id': best_config_id,
            'best_score': best_score,
            'all_results': self.results
        }
    
    def _load_and_harmonize(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and harmonize datasets"""
        # Load datasets
        df_ton = self.loader.load_ton_iot(
            sample_ratio=self.config.sample_ratio,
            random_state=self.config.random_state,
            incremental=False
        )
        
        df_cic = self.loader.load_cic_ddos2019(
            sample_ratio=self.config.sample_ratio,
            random_state=self.config.random_state,
            max_files_in_test=10 if self.config.test_mode else None
        )
        
        # Harmonize
        df_cic_harm, df_ton_harm = self.harmonizer.harmonize_features(df_cic, df_ton)
        
        return df_cic_harm, df_ton_harm
    
    def _early_fusion(self, df_cic: pd.DataFrame, df_ton: pd.DataFrame) -> pd.DataFrame:
        """Perform early fusion of harmonized datasets"""
        return self.harmonizer.early_fusion(df_cic, df_ton)
    
    def _evaluate_config(self, X: pd.DataFrame, y: pd.Series, 
                        preproc_config: Dict, config_id: int) -> float:
        """
        Evaluate a preprocessing configuration
        
        Args:
            X: Features
            y: Labels
            preproc_config: Preprocessing configuration dict
            config_id: Configuration ID
            
        Returns:
            Evaluation score (higher is better)
        """
        # Create preprocessing pipeline with config
        pipeline = PreprocessingPipeline(random_state=self.config.random_state)
        
        # Apply preprocessing steps according to config
        preprocessing_result = pipeline.prepare_data(
            X, y,
            apply_cleaning=preproc_config.get('apply_cleaning', True),
            apply_encoding=preproc_config.get('apply_encoding', True),
            apply_feature_selection=preproc_config.get('apply_feature_selection', False),
            apply_scaling=preproc_config.get('apply_scaling', True),
            apply_resampling=preproc_config.get('apply_resampling', False),
            apply_splitting=False  # No splitting in Phase 1 evaluation
        )
        
        # Simple evaluation: use a quick model (LogisticRegression) on a small sample
        # Score = accuracy on a quick train/test split
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        
        X_processed = preprocessing_result['X_processed']
        y_processed = preprocessing_result['y_processed']
        
        # Quick split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed,
            test_size=0.2,
            random_state=self.config.random_state,
            stratify=y_processed
        )
        
        # Train simple model
        model = LogisticRegression(random_state=self.config.random_state, max_iter=100)
        model.fit(X_train, y_train)
        
        # Evaluate
        score = model.score(X_test, y_test)
        
        return score
    
    def _save_results(self, best_config: Dict, best_config_id: int, best_score: float):
        """Save Phase 1 results"""
        # Save best config
        import json
        best_config_file = self.results_dir / 'best_config.json'
        with open(best_config_file, 'w') as f:
            json.dump({
                'config_id': best_config_id,
                'config': best_config,
                'score': best_score
            }, f, indent=2)
        
        # Save all results as CSV
        results_df = pd.DataFrame([
            {
                'config_id': r['config_id'],
                'score': r['score'],
                **r['config']
            }
            for r in self.results
        ])
        results_df = results_df.sort_values('score', ascending=False)
        results_file = self.results_dir / 'config_evaluation_results.csv'
        results_df.to_csv(results_file, index=False)
        
        logger.info(f"  Results saved to {self.results_dir}")


if __name__ == "__main__":
    # Test
    from ..config import TEST_CONFIG
    phase1 = Phase1ConfigSearch(TEST_CONFIG)
    print("✅ Phase 1 initialized")
