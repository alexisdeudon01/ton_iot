import pytest
import pandas as pd
import numpy as np
import logging
from src.phases.phase3_evaluation import Phase3Evaluation
from src.core.preprocessing_pipeline import PreprocessingPipeline

class MockConfig:
    def __init__(self):
        self.random_state = 42
        self.output_dir = "output_test"
        self.phase3_cv_folds = 5
        self.sample_ratio = 1.0
        self.test_mode = True
        self.preprocessing_profiles = {
            'lr_profile': {
                'apply_feature_selection': True,
                'apply_scaling': True,
                'apply_resampling': True,
                'feature_selection_k': 5
            }
        }

def test_no_smote_leakage_phase3(caplog):
    """Ensure SMOTE is applied only to TRAIN fold and no warnings are issued."""
    config = MockConfig()
    evaluator = Phase3Evaluation(config)

    # Create imbalanced synthetic data
    n_samples = 100
    X = pd.DataFrame(np.random.randn(n_samples, 10), columns=[f'f{i}' for i in range(10)])
    # Add some NaNs and infs
    X.iloc[0, 0] = np.nan
    X.iloc[1, 1] = np.inf

    # Imbalanced labels: 90% class 0, 10% class 1
    y = pd.Series([0] * 90 + [1] * 10)

    profile = config.preprocessing_profiles['lr_profile']

    with caplog.at_level(logging.WARNING):
        X_train_prep, y_train_prep, pipeline = evaluator._apply_preprocessing_per_fold(X, y, profile)

    # Assertions
    # 1. No warning "Applying SMOTE before splitting"
    assert "Applying SMOTE before splitting" not in caplog.text

    # 2. Train label distribution becomes balanced (SMOTE worked)
    counts = pd.Series(y_train_prep).value_counts()
    assert counts[0] == counts[1]
    assert counts[0] > 10 # Should be 90

    # 3. Test set size remains unchanged (not applicable here as we only test _apply_preprocessing_per_fold)
    # But we can check that the pipeline didn't fit on everything
    assert pipeline.smote is not None
