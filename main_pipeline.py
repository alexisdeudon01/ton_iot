#!/usr/bin/env python3
"""
Main pipeline for IRP research project
Orchestrates all phases:
- Phase 1: Preprocessing Configuration Selection
- Phase 3: Multi-Dimensional Algorithm Evaluation
- Phase 5: AHP-TOPSIS Ranking
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

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
    
    def __init__(self, results_dir: str = 'results', random_state: int = 42):
        """
        Initialize pipeline
        
        Args:
            results_dir: Directory for saving results
            random_state: Random seed for reproducibility
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        self.loader = DatasetLoader()
        self.harmonizer = DataHarmonizer()
        self.pipeline = PreprocessingPipeline(random_state=random_state)
        
        # Create result subdirectories
        (self.results_dir / 'phase1_preprocessing').mkdir(exist_ok=True)
        (self.results_dir / 'phase3_evaluation').mkdir(exist_ok=True)
        (self.results_dir / 'phase5_ranking').mkdir(exist_ok=True)
    
    def phase1_preprocessing(self) -> pd.DataFrame:
        """
        Phase 1: Preprocessing Configuration Selection
        - Load and harmonize datasets
        - Early fusion
        - Preprocessing with SMOTE and RobustScaler
        
        Returns:
            Preprocessed dataset
        """
        print("=" * 60)
        print("PHASE 1: Preprocessing Configuration Selection")
        print("=" * 60)
        
        # Load datasets
        print("\n1.1 Loading datasets...")
        try:
            df_ton = self.loader.load_ton_iot()
            print(f"   TON_IoT: {df_ton.shape}")
        except Exception as e:
            print(f"   Error loading TON_IoT: {e}")
            raise
        
        try:
            df_cic = self.loader.load_cic_ddos2019()
            print(f"   CIC-DDoS2019: {df_cic.shape}")
            
            # Harmonize and fuse
            print("\n1.2 Harmonizing datasets...")
            df_cic_harm, df_ton_harm = self.harmonizer.harmonize_features(df_cic, df_ton)
            print(f"   CIC-DDoS2019 harmonized: {df_cic_harm.shape}")
            print(f"   TON_IoT harmonized: {df_ton_harm.shape}")
            
            print("\n1.3 Early fusion...")
            df_fused, validation = self.harmonizer.early_fusion(df_cic_harm, df_ton_harm)
            print(f"   Fused dataset: {df_fused.shape}")
            print(f"   Validation: {sum(1 for v in validation.values() if v.get('compatible', False))}/{len(validation)} features compatible")
            
            # Use fused dataset
            df_processed = df_fused
            
        except FileNotFoundError:
            print("   CIC-DDoS2019 not available, using TON_IoT only")
            df_processed = df_ton
        
        # Prepare features and labels
        label_col = 'label' if 'label' in df_processed.columns else None
        if label_col is None:
            raise ValueError("Label column not found in dataset")
        
        X = df_processed.drop([label_col, 'dataset_source'] if 'dataset_source' in df_processed.columns else [label_col], axis=1, errors='ignore')
        y = df_processed[label_col]
        
        print("\n1.4 Preprocessing (SMOTE + RobustScaler)...")
        X_processed, y_processed = self.pipeline.prepare_data(
            X, y, apply_smote_flag=True, scale=True
        )
        print(f"   Processed shape: {X_processed.shape}")
        print(f"   Class distribution: {pd.Series(y_processed).value_counts().to_dict()}")
        
        # Save preprocessed data
        df_preprocessed = pd.DataFrame(X_processed)
        df_preprocessed['label'] = y_processed
        df_preprocessed.to_csv(self.results_dir / 'phase1_preprocessing' / 'preprocessed_data.csv', index=False)
        print(f"\n   Saved preprocessed data to {self.results_dir / 'phase1_preprocessing' / 'preprocessed_data.csv'}")
        
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
        
        print(f"\n3.1 Evaluating {len(models)} algorithms with 5-fold CV...")
        
        all_results = []
        
        for model_name, model in models.items():
            print(f"\n   Evaluating {model_name}...")
            
            fold_results = []
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                X_train_fold, X_test_fold = X[train_idx], X[test_idx]
                y_train_fold, y_test_fold = y[train_idx], y[test_idx]
                
                try:
                    results = evaluator.evaluate_model(
                        model, f"{model_name}_fold{fold_idx+1}",
                        X_train_fold, y_train_fold, X_test_fold, y_test_fold,
                        compute_explainability=False  # Disable for speed in CV
                    )
                    fold_results.append(results)
                except Exception as e:
                    print(f"      Error in fold {fold_idx+1}: {e}")
                    continue
            
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
                    'explainability_score': np.mean([r['explainability_score'] for r in fold_results])
                }
                all_results.append(avg_results)
                print(f"      F1 Score: {avg_results['f1_score']:.4f}, "
                      f"Time: {avg_results['training_time_seconds']:.2f}s")
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(self.results_dir / 'phase3_evaluation' / 'evaluation_results.csv', index=False)
        print(f"\n   Saved evaluation results to {self.results_dir / 'phase3_evaluation' / 'evaluation_results.csv'}")
        
        return results_df
    
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
    pipeline = IRPPipeline(results_dir='results', random_state=42)
    pipeline.run()


if __name__ == "__main__":
    main()
