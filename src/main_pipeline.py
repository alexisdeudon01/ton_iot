#!/usr/bin/env python3
"""
Main pipeline for IRP research project
Orchestrates all phases:
- Phase 1: Preprocessing Configuration Selection
- Phase 3: Multi-Dimensional Algorithm Evaluation
- Phase 5: AHP-TOPSIS Ranking
"""
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Setup logger
logger = logging.getLogger(__name__)

# Import custom modules
from dataset_loader import DatasetLoader
from data_harmonization import DataHarmonizer
from preprocessing_pipeline import PreprocessingPipeline, StratifiedCrossValidator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from models_cnn import CNNTabularClassifier
from models_tabnet import TabNetClassifierWrapper
from evaluation_3d import Evaluation3D
from ahp_topsis_framework import AHPTopsisFramework

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class IRPPipeline:
    """Complete IRP research pipeline"""
    
    def __init__(self, results_dir: str = 'output', random_state: int = 42):
        """
        Initialize pipeline
        
        Args:
            results_dir: Directory for saving results (default: 'output')
            random_state: Random seed for reproducibility
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.loader = DatasetLoader()
        self.harmonizer = DataHarmonizer()
        self.pipeline = PreprocessingPipeline(random_state=random_state)
        
        # Create result subdirectories
        (self.results_dir / 'phase1_preprocessing').mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'phase3_evaluation').mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'phase3_evaluation' / 'algorithm_reports').mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'phase3_evaluation' / 'visualizations').mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'phase5_ranking').mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'logs').mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline initialized with output directory: {self.results_dir.absolute()}")
    
    def phase1_preprocessing(self) -> pd.DataFrame:
        """
        Phase 1: Preprocessing Configuration Selection
        - Load and harmonize datasets (TON_IoT + CIC-DDoS2019)
        - Early fusion with statistical validation (Kolmogorov-Smirnov)
        - Preprocessing with SMOTE (class balancing) and RobustScaler (feature scaling)
        
        Returns:
            Tuple of (X_processed, y_processed, feature_names)
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: Preprocessing Configuration Selection")
        logger.info("=" * 60)
        
        # Load datasets
        logger.info("\n1.1 Loading datasets...")
        try:
            df_ton = self.loader.load_ton_iot()
            logger.info(f"   TON_IoT: {df_ton.shape}")
        except FileNotFoundError as e:
            logger.error(f"   Error loading TON_IoT: {e}")
            raise
        except Exception as e:
            logger.error(f"   Unexpected error loading TON_IoT: {e}", exc_info=True)
            raise
        
        try:
            df_cic = self.loader.load_cic_ddos2019()
            logger.info(f"   CIC-DDoS2019: {df_cic.shape}")
            
            # Harmonize and fuse
            logger.info("\n1.2 Harmonizing datasets...")
            logger.info("   Mapping features between TON_IoT and CIC-DDoS2019...")
            try:
                df_cic_harm, df_ton_harm = self.harmonizer.harmonize_features(df_cic, df_ton)
                logger.info(f"   CIC-DDoS2019 harmonized: {df_cic_harm.shape}")
                logger.info(f"   TON_IoT harmonized: {df_ton_harm.shape}")
            except Exception as e:
                logger.error(f"   Error during harmonization: {e}", exc_info=True)
                raise
            
            logger.info("\n1.3 Early fusion...")
            logger.info("   Performing statistical validation (Kolmogorov-Smirnov test)...")
            try:
                df_fused, validation = self.harmonizer.early_fusion(df_cic_harm, df_ton_harm)
                compatible_count = sum(1 for v in validation.values() if v.get('compatible', False))
                logger.info(f"   Fused dataset: {df_fused.shape}")
                logger.info(f"   Validation: {compatible_count}/{len(validation)} features compatible")
            except Exception as e:
                logger.error(f"   Error during early fusion: {e}", exc_info=True)
                raise
            
            # Use fused dataset
            df_processed = df_fused
            
        except FileNotFoundError:
            logger.warning("   CIC-DDoS2019 not available, using TON_IoT only")
            df_processed = df_ton
        except Exception as e:
            logger.error(f"   Error processing CIC-DDoS2019: {e}", exc_info=True)
            logger.warning("   Falling back to TON_IoT only")
            df_processed = df_ton
        
        # Prepare features and labels
        label_col = 'label' if 'label' in df_processed.columns else None
        if label_col is None:
            error_msg = "Label column not found in dataset"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            X = df_processed.drop([label_col, 'dataset_source'] if 'dataset_source' in df_processed.columns else [label_col], axis=1, errors='ignore')
            y = df_processed[label_col]
            logger.info(f"   Features shape: {X.shape}, Labels shape: {y.shape}")
        except Exception as e:
            logger.error(f"   Error preparing features/labels: {e}", exc_info=True)
            raise
        
        logger.info("\n1.4 Preprocessing (SMOTE + RobustScaler)...")
        logger.info("   Applying SMOTE for class balancing...")
        logger.info("   Scaling features with RobustScaler...")
        try:
            X_processed, y_processed = self.pipeline.prepare_data(
                X, y, apply_smote_flag=True, scale=True
            )
            logger.info(f"   Processed shape: {X_processed.shape}")
            class_dist = pd.Series(y_processed).value_counts().to_dict()
            logger.info(f"   Class distribution: {class_dist}")
        except MemoryError as e:
            logger.error(f"   Memory error during preprocessing: {e}")
            logger.error("   Try reducing dataset size or increasing available memory")
            raise
        except Exception as e:
            logger.error(f"   Error during preprocessing: {e}", exc_info=True)
            raise
        
        # Save preprocessed data
        try:
            df_preprocessed = pd.DataFrame(X_processed)
            df_preprocessed['label'] = y_processed
            output_file = self.results_dir / 'phase1_preprocessing' / 'preprocessed_data.csv'
            df_preprocessed.to_csv(output_file, index=False)
            logger.info(f"\n   Saved preprocessed data to {output_file}")
        except Exception as e:
            logger.error(f"   Error saving preprocessed data: {e}", exc_info=True)
            raise
        
        return X_processed, y_processed, self.pipeline.feature_names
    
    def phase3_evaluation(self, X: np.ndarray, y: np.ndarray, feature_names: list = None):
        """
        Phase 3: Multi-Dimensional Algorithm Evaluation
        - Train and evaluate 5 algorithms
        - Evaluate across 3 dimensions
        
        Returns:
            Evaluation results DataFrame
        """
        print("\n" + "=" * 60)
        print("PHASE 3: Multi-Dimensional Algorithm Evaluation")
        print("=" * 60)
        
        # Initialize models (according to IRP methodology)
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'CNN': CNNTabularClassifier(epochs=20, batch_size=64, random_state=self.random_state),
        }
        
        # Add TabNet if available
        try:
            models['TabNet'] = TabNetClassifierWrapper(max_epochs=50, batch_size=1024, seed=self.random_state, verbose=0)
        except ImportError:
            print("   Warning: TabNet not available (pytorch-tabnet not installed)")
        
        # Use stratified cross-validation
        cv = StratifiedCrossValidator(n_splits=5, random_state=self.random_state)
        
        # Initialize 3D evaluator
        feature_names_list = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        evaluator = Evaluation3D(feature_names=feature_names_list)
        
        logger.info(f"\n3.1 Evaluating {len(models)} algorithms with 5-fold CV...")
        
        all_results = []
        
        for model_name, model_template in tqdm(models.items(), desc="Evaluating algorithms"):
            logger.info(f"\n   Evaluating {model_name}...")
            
            fold_results = []
            fold_level_results = []  # Store per-fold results for debugging
            
            for fold_idx, (train_idx, test_idx) in enumerate(tqdm(cv.split(X, y), desc=f"  {model_name} CV folds", total=5, leave=False)):
                X_train_fold, X_test_fold = X[train_idx], X[test_idx]
                y_train_fold, y_test_fold = y[train_idx], y[test_idx]
                
                try:
                    # Create a fresh model instance for each fold to avoid state contamination
                    from sklearn.base import clone
                    try:
                        model = clone(model_template)
                    except (TypeError, AttributeError) as clone_err:
                        # For custom models that can't be cloned, reinitialize if possible
                        logger.debug(f"      Could not clone {model_name}, using template: {clone_err}")
                        model = model_template
                    
                    # Ensure model is fitted before evaluation (evaluate_model handles this internally)
                    results = evaluator.evaluate_model(
                        model, f"{model_name}_fold{fold_idx+1}",
                        X_train_fold, y_train_fold, X_test_fold, y_test_fold,
                        compute_explainability=False  # Disable for speed in CV
                    )
                    
                    # Validate that required metrics are present
                    required_keys = ['f1_score', 'accuracy', 'precision', 'recall', 
                                   'training_time_seconds', 'memory_used_mb', 'explainability_score']
                    missing_keys = [k for k in required_keys if k not in results]
                    if missing_keys:
                        raise ValueError(f"Missing required metrics: {missing_keys}")
                    
                    # Add fold information
                    results['fold'] = fold_idx + 1
                    fold_results.append(results)
                    fold_level_results.append({
                        'model_name': model_name,
                        'fold': fold_idx + 1,
                        **{k: v for k, v in results.items() if k in required_keys}
                    })
                    
                    logger.debug(f"      Fold {fold_idx+1} - F1: {results['f1_score']:.4f}, "
                               f"Acc: {results['accuracy']:.4f}")
                    
                except Exception as e:
                    logger.warning(f"      Error in fold {fold_idx+1} for {model_name}: {e}")
                    logger.debug(f"      Full error traceback:", exc_info=True)
                    continue
            
            # Only add averaged results if we have at least one successful fold
            if fold_results:
                # Average results across folds
                avg_results = {
                    'model_name': model_name,
                    'f1_score': np.mean([r['f1_score'] for r in fold_results]),
                    'accuracy': np.mean([r['accuracy'] for r in fold_results]),
                    'precision': np.mean([r['precision'] for r in fold_results]),
                    'recall': np.mean([r['recall'] for r in fold_results]),
                    'training_time_seconds': np.mean([r['training_time_seconds'] for r in fold_results]),
                    'memory_used_mb': np.mean([r['memory_used_mb'] for r in fold_results]),
                    'explainability_score': np.mean([r['explainability_score'] for r in fold_results]),
                    'n_successful_folds': len(fold_results)  # Track how many folds succeeded
                }
                all_results.append(avg_results)
                logger.info(f"      {model_name} - F1 Score: {avg_results['f1_score']:.4f}, "
                           f"Time: {avg_results['training_time_seconds']:.2f}s, "
                           f"Explainability: {avg_results['explainability_score']:.4f}, "
                           f"Successful folds: {len(fold_results)}/5")
            else:
                logger.error(f"      {model_name} - All folds failed! Model not evaluated.")
        
        # Create results DataFrame - ensure it's never empty
        try:
            if not all_results:
                logger.error("   CRITICAL: No evaluation results collected! Creating empty structure.")
                # Create empty DataFrame with required columns to prevent downstream failures
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
                logger.error("   Check logs above for errors preventing model evaluation.")
            
            results_df = pd.DataFrame(all_results)
            
            # Validate required columns are present
            required_columns = ['model_name', 'f1_score', 'accuracy', 'precision', 'recall']
            missing_cols = [col for col in required_columns if col not in results_df.columns]
            if missing_cols:
                logger.error(f"   Missing required columns in results: {missing_cols}")
                raise ValueError(f"Results DataFrame missing required columns: {missing_cols}")
            
            output_file = self.results_dir / 'phase3_evaluation' / 'evaluation_results.csv'
            results_df.to_csv(output_file, index=False)
            logger.info(f"\n   Saved evaluation results to {output_file}")
            logger.info(f"   Total models evaluated: {len(results_df)}")
        except Exception as e:
            logger.error(f"   Error saving evaluation results: {e}", exc_info=True)
            raise
        
        # Generate dimension scores
        try:
            dimension_scores = evaluator.get_dimension_scores()
            dimension_scores_file = self.results_dir / 'phase3_evaluation' / 'dimension_scores.csv'
            dimension_scores.to_csv(dimension_scores_file, index=False)
            logger.info(f"   Saved dimension scores to {dimension_scores_file}")
        except Exception as e:
            logger.warning(f"   Error generating dimension scores: {e}")
        
        # Generate algorithm reports
        logger.info("\n3.2 Generating algorithm reports...")
        try:
            for _, row in results_df.iterrows():
                results_dict = row.to_dict()
                evaluator.generate_algorithm_report(
                    results_dict['model_name'], 
                    results_dict,
                    self.results_dir / 'phase3_evaluation'
                )
            logger.info("   Algorithm reports generated")
        except Exception as e:
            logger.warning(f"   Error generating algorithm reports: {e}")
        
        # Generate visualizations (if matplotlib available)
        logger.info("\n3.3 Generating visualizations...")
        try:
            self._generate_visualizations(results_df, evaluator)
            logger.info("   Visualizations generated")
        except Exception as e:
            logger.warning(f"   Error generating visualizations: {e}")
        
        return results_df
    
    def _generate_visualizations(self, results_df: pd.DataFrame, evaluator):
        """Generate visualizations for the 3 dimensions"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            viz_dir = self.results_dir / 'phase3_evaluation' / 'visualizations'
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Dimension 1: Performance metrics
            fig, ax = plt.subplots(figsize=(10, 6))
            x = range(len(results_df))
            width = 0.2
            ax.bar([i - width for i in x], results_df['f1_score'], width, label='F1 Score', color='#1f77b4')
            ax.bar(x, results_df['precision'], width, label='Precision', color='#ff7f0e')
            ax.bar([i + width for i in x], results_df['recall'], width, label='Recall', color='#2ca02c')
            ax.set_xlabel('Algorithm')
            ax.set_ylabel('Score')
            ax.set_title('Dimension 1: Detection Performance')
            ax.set_xticks(x)
            ax.set_xticklabels(results_df['model_name'], rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(viz_dir / 'dimension1_performance.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Dimension 2: Resource Efficiency
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            ax1.bar(results_df['model_name'], results_df['training_time_seconds'], color='#d62728')
            ax1.set_ylabel('Training Time (seconds)')
            ax1.set_title('Training Time by Algorithm')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis='y', alpha=0.3)
            
            ax2.bar(results_df['model_name'], results_df['memory_used_mb'], color='#9467bd')
            ax2.set_ylabel('Memory Usage (MB)')
            ax2.set_title('Memory Usage by Algorithm')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(viz_dir / 'dimension2_resources.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Dimension 3: Explainability
            fig, ax = plt.subplots(figsize=(10, 6))
            x = range(len(results_df))
            width = 0.25
            ax.bar([i - width for i in x], results_df['native_interpretability'], width, label='Native', color='#8c564b')
            if 'shap_score' in results_df.columns:
                shap_values = results_df['shap_score'].fillna(0)
                ax.bar(x, shap_values, width, label='SHAP', color='#e377c2')
            if 'lime_score' in results_df.columns:
                lime_values = results_df['lime_score'].fillna(0)
                ax.bar([i + width for i in x], lime_values, width, label='LIME', color='#7f7f7f')
            ax.set_xlabel('Algorithm')
            ax.set_ylabel('Score')
            ax.set_title('Dimension 3: Explainability')
            ax.set_xticks(x)
            ax.set_xticklabels(results_df['model_name'], rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(viz_dir / 'dimension3_explainability.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Combined radar chart (if dimension scores available)
            try:
                dimension_scores = evaluator.get_dimension_scores()
                fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
                categories = ['Detection\nPerformance', 'Resource\nEfficiency', 'Explainability']
                num_vars = len(categories)
                angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
                angles += angles[:1]
                
                for idx, row in dimension_scores.iterrows():
                    values = [
                        row['detection_performance'],
                        row['resource_efficiency'],
                        row['explainability']
                    ]
                    values += values[:1]
                    ax.plot(angles, values, 'o-', linewidth=2, label=row['model_name'])
                    ax.fill(angles, values, alpha=0.1)
                
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
                ax.set_ylim(0, 1)
                ax.set_title('3D Evaluation: Combined Dimensions', pad=20)
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                ax.grid(True)
                plt.tight_layout()
                plt.savefig(viz_dir / 'combined_3d_radar.png', dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                logger.debug(f"Could not generate radar chart: {e}")
                
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available. Skipping visualizations.")
        except Exception as e:
            logger.warning(f"Error generating visualizations: {e}", exc_info=True)
    
    def phase5_ranking(self, evaluation_results: pd.DataFrame):
        """
        Phase 5: AHP-TOPSIS Ranking
        
        Args:
            evaluation_results: DataFrame with evaluation results from Phase 3
            
        Returns:
            Ranking results DataFrame
        """
        print("\n" + "=" * 60)
        print("PHASE 5: AHP-TOPSIS Ranking")
        print("=" * 60)
        
        # Define criteria
        criteria = ['Detection Performance', 'Resource Efficiency', 'Explainability']
        
        # Initialize framework
        framework = AHPTopsisFramework(
            criteria_names=criteria,
            criteria_types=['max', 'max', 'max']
        )
        
        # Set AHP comparisons (example - can be adjusted)
        print("\n5.1 Setting AHP pairwise comparisons...")
        comparisons = {
            ('Detection Performance', 'Resource Efficiency'): 3,  # Detection Performance moderately more important
            ('Detection Performance', 'Explainability'): 2,  # Detection Performance slightly more important
            ('Resource Efficiency', 'Explainability'): 0.5  # Explainability slightly more important
        }
        framework.set_ahp_comparisons(comparisons)
        
        print("\n5.2 AHP Weights:")
        weights_df = framework.get_weights()
        print(weights_df)
        weights_df.to_csv(self.results_dir / 'phase5_ranking' / 'ahp_weights.csv', index=False)
        
        # Prepare decision matrix from evaluation results
        print("\n5.3 Preparing decision matrix...")
        decision_matrix = np.array([
            evaluation_results['f1_score'].values,  # Dimension 1: Detection Performance
            1 / (1 + evaluation_results['training_time_seconds'].values / evaluation_results['training_time_seconds'].max()),  # Dimension 2: Resource Efficiency (normalized, higher is better)
            evaluation_results['explainability_score'].values  # Dimension 3: Explainability
        ]).T
        
        # Normalize to [0, 1] scale
        for j in range(decision_matrix.shape[1]):
            col = decision_matrix[:, j]
            col_min, col_max = col.min(), col.max()
            if col_max > col_min:
                decision_matrix[:, j] = (col - col_min) / (col_max - col_min)
        
        framework.set_decision_matrix(decision_matrix, evaluation_results['model_name'].tolist())
        
        # Perform ranking
        print("\n5.4 Computing TOPSIS ranking...")
        ranking_results = framework.rank_alternatives()
        
        print("\n5.5 Final Ranking:")
        print(ranking_results)
        
        ranking_results.to_csv(self.results_dir / 'phase5_ranking' / 'ranking_results.csv', index=False)
        print(f"\n   Saved ranking results to {self.results_dir / 'phase5_ranking' / 'ranking_results.csv'}")
        
        return ranking_results
    
    def run(self):
        """Run complete pipeline"""
        print("\n" + "=" * 60)
        print("IRP RESEARCH PIPELINE")
        print("AI-Powered Log Analysis for Smarter Threat Detection")
        print("=" * 60)
        
        try:
            # Phase 1: Preprocessing
            X, y, feature_names = self.phase1_preprocessing()
            
            # Phase 3: Evaluation
            evaluation_results = self.phase3_evaluation(X, y, feature_names)
            
            # Phase 5: Ranking
            ranking_results = self.phase5_ranking(evaluation_results)
            
            print("\n" + "=" * 60)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"\nResults saved in: {self.results_dir}")
            print("  - Phase 1: preprocessing/")
            print("  - Phase 3: evaluation/")
            print("  - Phase 5: ranking/")
            
        except Exception as e:
            print(f"\nError in pipeline: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Run the main pipeline"""
    pipeline = IRPPipeline(results_dir='output', random_state=42)
    pipeline.run()


if __name__ == "__main__":
    main()
