import numpy as np
import pandas as pd
import pytest

from src.core.preprocessing_pipeline import PreprocessingPipeline

pytest.importorskip("imblearn")


def test_no_data_leakage_transform_test():
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

    assert X_train_processed.shape[0] >= X_train.shape[0]
    assert X_test_processed.shape[0] == X_test.shape[0]
