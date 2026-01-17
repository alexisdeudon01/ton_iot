#!/usr/bin/env python3
"""Tests to ensure preprocessing fits only on training data."""
import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.phases.phase3_evaluation import Phase3Evaluation


def test_preprocessing_fit_uses_train_only():
    """Ensure scaler/imputer stats match training data, not train+test."""
    config = PipelineConfig()
    phase3 = Phase3Evaluation(config)

    X_train = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0],
        "f2": [10.0, 20.0, 30.0, 40.0],
    })
    y_train = pd.Series([0, 1, 0, 1])

    X_test = pd.DataFrame({
        "f1": [1000.0],
        "f2": [-999.0],
    })

    profile = {
        "apply_feature_selection": False,
        "apply_scaling": True,
        "apply_resampling": False,
    }

    _, _, pipeline = phase3._apply_preprocessing_per_fold(X_train, y_train, profile)

    expected_train_median = np.median(X_train.values, axis=0)
    combined_median = np.median(np.vstack([X_train.values, X_test.values]), axis=0)

    assert np.allclose(pipeline.imputer.statistics_, expected_train_median)
    assert np.allclose(pipeline.scaler.center_, expected_train_median)
    assert not np.allclose(pipeline.scaler.center_, combined_median)

    transformed = pipeline.transform_test(X_test.values)
    assert transformed.shape == X_test.values.shape
