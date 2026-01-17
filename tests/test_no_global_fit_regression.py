import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.phases.phase3_evaluation import Phase3Evaluation
from src.core.preprocessing_pipeline import PreprocessingPipeline

class MockConfig:
    def __init__(self):
        self.random_state = 42
        self.output_dir = "output_test"
        self.phase3_cv_folds = 2
        self.sample_ratio = 1.0
        self.test_mode = True
        self.synthetic_mode = True
        self.phase3_algorithms = ['logistic-regression']
        self.preprocessing_profiles = {
            'lr_profile': {
                'apply_feature_selection': True,
                'apply_scaling': True,
                'apply_resampling': True,
                'feature_selection_k': 5
            }
        }

def test_no_global_fit_regression():
    """Ensure no fit() is called on preprocessing objects using full dataset before CV."""
    config = MockConfig()
    evaluator = Phase3Evaluation(config)

    # Mock the entire PreprocessingPipeline to track calls
    with patch('src.phases.phase3_evaluation.PreprocessingPipeline') as MockPipeline:
        # Setup the mock instance
        mock_pipeline_inst = MockPipeline.return_value
        mock_pipeline_inst.prepare_data.return_value = {
            'X_processed': np.zeros((10, 5)),
            'y_processed': np.zeros(10),
            'feature_names': ['f1', 'f2', 'f3', 'f4', 'f5']
        }
        mock_pipeline_inst.transform_test.return_value = np.zeros((5, 5))
        mock_pipeline_inst.selected_features = ['f1', 'f2', 'f3', 'f4', 'f5']

        # Mock _load_and_prepare_dataset to return a small df
        X_rand = np.random.randn(20, 5)
        y_rand = np.array([0] * 10 + [1] * 10) # Discrete labels for StratifiedKFold
        df = pd.DataFrame(X_rand, columns=[f'f{i}' for i in range(5)])
        df['label'] = y_rand
        evaluator._load_and_prepare_dataset = MagicMock(return_value=df)

        # Mock Evaluation3D to avoid actual evaluation
        with patch('src.phases.phase3_evaluation.Evaluation3D') as MockEval3D:
            mock_eval_inst = MockEval3D.return_value
            mock_eval_inst.evaluate_model.return_value = {
                'f1_score': 0.9, 'accuracy': 0.9, 'precision': 0.9, 'recall': 0.9,
                'training_time_seconds': 0.1, 'memory_used_mb': 10, 'explainability_score': 0.8
            }
            mock_eval_inst.get_dimension_scores.return_value = pd.DataFrame()

            # Run Phase 3
            evaluator.run()

            # Verify that PreprocessingPipeline was instantiated inside the loop
            # (once per fold per model)
            # In our mock config: 1 model * 2 folds = 2 instantiations
            assert MockPipeline.call_count == 2

            # Verify that no global pipeline was used before the loop
            # We can check if any other PreprocessingPipeline was created
            # (Phase3Evaluation doesn't have one in __init__)
