"""
Tests for data leakage prevention in preprocessing
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.core.preprocessing_pipeline import PreprocessingPipeline

pytest.importorskip("imblearn", reason="imblearn not available")


def test_no_data_leakage_transform_test():
    """Test that transform_test() prevents data leakage (fits only on training data)"""
    rng = np.random.default_rng(42)
    X_train = pd.DataFrame(
        {
            "f1": rng.normal(size=50),
            "f2": rng.normal(size=50),
        }
    )
    y_train = pd.Series([0] * 45 + [1] * 5)
    X_test = pd.DataFrame(
        {
            "f1": rng.normal(size=10),
            "f2": rng.normal(size=10),
        }
    )

    pipeline = PreprocessingPipeline(random_state=42, n_features=2)
    result = pipeline.prepare_data(
        X_train,
        y_train,
        apply_encoding=False,
        apply_feature_selection=True,
        apply_scaling=True,
        apply_resampling=True,
        apply_splitting=False,
    )

    X_train_processed = result["X_processed"]
    X_test_processed = pipeline.transform_test(X_test.values)

    assert X_train_processed.shape[0] >= X_train.shape[0], \
        f"Training data may have been resampled (original: {X_train.shape[0]}, processed: {X_train_processed.shape[0]})"
    assert X_test_processed.shape[0] == X_test.shape[0], \
        f"Test data shape should be preserved (expected: {X_test.shape[0]}, got: {X_test_processed.shape[0]})"
    assert X_test_processed.shape[1] == X_train_processed.shape[1], \
        f"Test data should have same number of features as training (train: {X_train_processed.shape[1]}, test: {X_test_processed.shape[1]})"
