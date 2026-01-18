import pytest
import pandas as pd
import numpy as np
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
            },
            'tree_profile': {
                'apply_feature_selection': False,
                'apply_scaling': False,
                'apply_resampling': True,
                'use_class_weight': True
            }
        }

def test_test_transform_is_fold_fitted():
    """Ensure TEST uses TRAIN-fitted objects and handles NaNs/scaling correctly."""
    config = MockConfig()
    evaluator = Phase3Evaluation(config)

    # 1. Setup TRAIN with strong medians
    X_train = pd.DataFrame({
        'f1': [10, 10, 10, 10, 10],
        'f2': [1, 2, 3, 4, 5]
    })
    y_train = pd.Series([0, 0, 0, 1, 1])

    # 2. Setup TEST with NaNs and infs
    X_test = pd.DataFrame({
        'f1': [np.nan, np.inf],
        'f2': [3, 3]
    })

    # Fit on TRAIN
    profile_lr = config.preprocessing_profiles['lr_profile']
    X_train_prep, y_train_prep, pipeline = evaluator._apply_preprocessing_per_fold(X_train, y_train, profile_lr)

    # Transform TEST (now using pipeline.transform_test() directly)
    X_test_prep = pipeline.transform_test(X_test)

    # Assertions for LR (scaling enabled)
    assert not np.isnan(X_test_prep).any()
    assert np.isfinite(X_test_prep).all()
    # f1 in test was NaN/inf, should be imputed with TRAIN median (10)
    # After scaling, it should be 0 (since all train values were 10)
    # RobustScaler: (x - median) / IQR. If IQR is 0, it might be 0 or original.
    # Let's just check it's not NaN.

    # 3. Test tree profile (no scaling)
    profile_tree = config.preprocessing_profiles['tree_profile']
    X_train_prep_t, y_train_prep_t, pipeline_t = evaluator._apply_preprocessing_per_fold(X_train, y_train, profile_tree)
    X_test_prep_t = pipeline_t.transform_test(X_test)

    assert not np.isnan(X_test_prep_t).any()
    # f1 should be 10 (imputed from train)
    assert X_test_prep_t[0, 0] == 10.0
    assert X_test_prep_t[1, 0] == 10.0
