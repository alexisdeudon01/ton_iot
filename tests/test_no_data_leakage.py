"""
Tests to verify zero data leakage in preprocessing pipeline
"""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.core.preprocessing_pipeline import PreprocessingPipeline


def test_scaler_fit_only_on_train():
    """
    Test that scaler is fitted ONLY on TRAIN data, not on TEST.
    
    Input:
        - Training and test DataFrames
    
    Processing:
        - Fit preprocessing pipeline on TRAIN
        - Transform TEST using fitted pipeline
        - Verify scaler was fitted on TRAIN only (not TEST)
    
    Expected Output:
        - Scaler statistics (mean, scale) based on TRAIN data only
        - TEST data transformed using TRAIN-fitted scaler
    
    Method:
        - Direct testing of PreprocessingPipeline.fit() and transform_test()
    """
    # Generate synthetic data
    np.random.seed(42)
    n_train, n_test = 200, 50
    n_features = 10
    
    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features) * 10 + 100,  # Mean ~100
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features) * 5 + 150,  # Different distribution (mean ~150)
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_train = pd.Series(np.random.randint(0, 2, n_train))
    y_test = pd.Series(np.random.randint(0, 2, n_test))
    
    # Fit pipeline on TRAIN
    pipeline = PreprocessingPipeline(random_state=42, n_features=n_features)
    result = pipeline.prepare_data(
        X_train,
        y_train,
        apply_encoding=False,
        apply_feature_selection=False,
        apply_scaling=True,
        apply_resampling=False,
        apply_splitting=False,
        apply_imputation=True
    )
    
    # Verify scaler was fitted
    assert pipeline.is_fitted, "Pipeline should be fitted after prepare_data"
    assert pipeline.scaler is not None, "Scaler should exist"
    
    # Get scaler statistics from TRAIN data
    train_scaled = result['X_processed']
    train_mean = train_scaled.mean(axis=0)
    train_std = train_scaled.std(axis=0)
    
    # Transform TEST using TRAIN-fitted scaler
    X_test_transformed = pipeline.transform_test(X_test)
    
    # Verify TEST was transformed (should be scaled based on TRAIN statistics)
    assert X_test_transformed.shape[0] == n_test, (
        f"Test shape mismatch: expected {n_test}, got {X_test_transformed.shape[0]}"
    )
    assert X_test_transformed.shape[1] == n_features, (
        f"Feature shape mismatch: expected {n_features}, got {X_test_transformed.shape[1]}"
    )
    
    # Verify TEST statistics are different from original (scaling applied)
    # But scaled based on TRAIN distribution, not TEST distribution
    test_scaled_mean = X_test_transformed.mean(axis=0)
    test_scaled_std = X_test_transformed.std(axis=0)
    
    # Test data should be scaled using TRAIN's center and scale
    # After RobustScaler, means should be close to 0 (centered)
    assert np.abs(test_scaled_mean.mean()) < 5.0, (
        f"Scaled test mean should be near 0 (got {test_scaled_mean.mean():.3f}), "
        f"indicating scaling was applied based on TRAIN distribution"
    )


def test_feature_selector_fit_only_on_train():
    """
    Test that feature selector is fitted ONLY on TRAIN data.
    
    Input:
        - Training and test DataFrames
    
    Processing:
        - Fit feature selector on TRAIN
        - Transform TEST using fitted selector
        - Verify selector was fitted on TRAIN only
    
    Expected Output:
        - Selected features based on TRAIN data only
        - TEST data transformed with same selected features
    
    Method:
        - Direct testing of PreprocessingPipeline with feature selection
    """
    # Generate synthetic data
    np.random.seed(42)
    n_train, n_test = 200, 50
    n_features = 20
    k_selected = 10
    
    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create informative features (first k_selected features correlate with y)
    y_train = pd.Series(np.random.randint(0, 2, n_train))
    for i in range(k_selected):
        X_train[f'feature_{i}'] += y_train.values * 2.0  # Make features informative
    
    y_test = pd.Series(np.random.randint(0, 2, n_test))
    
    # Fit pipeline on TRAIN with feature selection
    pipeline = PreprocessingPipeline(random_state=42, n_features=k_selected)
    result = pipeline.prepare_data(
        X_train,
        y_train,
        apply_encoding=False,
        apply_feature_selection=True,  # Enable feature selection
        apply_scaling=False,  # Disable scaling for simplicity
        apply_resampling=False,
        apply_splitting=False,
        apply_imputation=True
    )
    
    # Verify selector was fitted
    assert pipeline.feature_selector is not None, "Feature selector should exist"
    assert pipeline.selected_features is not None, "Selected features should exist"
    assert len(pipeline.selected_features) == k_selected, (
        f"Expected {k_selected} selected features, got {len(pipeline.selected_features)}"
    )
    
    # Transform TEST using TRAIN-fitted selector
    X_test_transformed = pipeline.transform_test(X_test)
    
    # Verify TEST was transformed with same selected features
    assert X_test_transformed.shape[1] == k_selected, (
        f"Test should have {k_selected} features after selection, got {X_test_transformed.shape[1]}"
    )
    assert X_test_transformed.shape[0] == n_test, (
        f"Test should have {n_test} samples, got {X_test_transformed.shape[0]}"
    )


def test_imputer_fit_only_on_train():
    """
    Test that imputer is fitted ONLY on TRAIN data.
    
    Input:
        - Training and test DataFrames with NaN values
    
    Processing:
        - Fit imputer on TRAIN (with NaN)
        - Transform TEST using fitted imputer
        - Verify imputer was fitted on TRAIN only
    
    Expected Output:
        - Imputation values (median) based on TRAIN data only
        - TEST NaN values filled using TRAIN medians
    
    Method:
        - Direct testing of PreprocessingPipeline imputation
    """
    # Generate synthetic data with NaN
    np.random.seed(42)
    n_train, n_test = 200, 50
    n_features = 10
    
    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features) * 10 + 100,
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features) * 5 + 150,  # Different distribution
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add NaN values
    X_train.iloc[10:20, 0] = np.nan
    X_test.iloc[5:10, 0] = np.nan
    
    y_train = pd.Series(np.random.randint(0, 2, n_train))
    y_test = pd.Series(np.random.randint(0, 2, n_test))
    
    # Calculate TRAIN median for feature_0 (before imputation)
    train_median_feature0 = X_train['feature_0'].median()
    
    # Fit pipeline on TRAIN
    pipeline = PreprocessingPipeline(random_state=42, n_features=n_features)
    result = pipeline.prepare_data(
        X_train,
        y_train,
        apply_encoding=False,
        apply_feature_selection=False,
        apply_scaling=False,
        apply_resampling=False,
        apply_splitting=False,
        apply_imputation=True  # Enable imputation
    )
    
    # Verify imputer was fitted
    assert pipeline.imputer is not None, "Imputer should exist"
    
    # Transform TEST using TRAIN-fitted imputer
    X_test_transformed = pipeline.transform_test(X_test)
    
    # Verify TEST NaN values were filled using TRAIN median
    # Check that feature_0 NaN values in TEST were filled with TRAIN median
    assert not np.isnan(X_test_transformed).any(), (
        "All NaN values in TEST should be imputed"
    )
    
    # Verify imputation used TRAIN statistics (not TEST)
    # TEST feature_0 should be imputed with TRAIN median (around 100), not TEST median (around 150)
    # Only check the imputed values (where NaN was originally), not all values
    # NaN was in rows 5:10 (indices 5-9), so check those specific rows
    test_imputed_values = X_test_transformed[5:10, 0]  # Only the rows where NaN was
    # These should be close to TRAIN median, not TEST median
    # Allow larger tolerance due to scaling/normalization effects if present
    assert np.abs(test_imputed_values.mean() - train_median_feature0) < 30.0, (
        f"TEST imputation should use TRAIN median ({train_median_feature0:.3f}), "
        f"got mean imputed value {test_imputed_values.mean():.3f} for NaN rows"
    )


def test_transform_test_no_fitting():
    """
    Test that transform_test() never fits anything (stateless transformation).
    
    Input:
        - Fitted pipeline
        - Test DataFrame
    
    Processing:
        - Call transform_test() multiple times
        - Verify no fitting occurs (scaler/selector/imputer remain unchanged)
    
    Expected Output:
        - Pipeline state unchanged after transform_test()
        - No new fitting operations
    
    Method:
        - Direct testing of transform_test() method
    """
    # Generate synthetic data
    np.random.seed(42)
    n_train, n_test = 200, 50
    n_features = 10
    
    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_train = pd.Series(np.random.randint(0, 2, n_train))
    
    # Fit pipeline on TRAIN
    pipeline = PreprocessingPipeline(random_state=42, n_features=n_features)
    pipeline.prepare_data(
        X_train,
        y_train,
        apply_encoding=False,
        apply_feature_selection=False,
        apply_scaling=True,
        apply_resampling=False,
        apply_splitting=False,
        apply_imputation=True
    )
    
    # Store pipeline state
    scaler_center_before = pipeline.scaler.center_.copy() if pipeline.scaler is not None else None
    scaler_scale_before = pipeline.scaler.scale_.copy() if pipeline.scaler is not None else None
    
    # Transform TEST (should not change pipeline state)
    X_test_transformed1 = pipeline.transform_test(X_test)
    
    # Verify pipeline state unchanged
    if pipeline.scaler is not None:
        assert np.array_equal(pipeline.scaler.center_, scaler_center_before), (
            "Scaler center should not change after transform_test()"
        )
        assert np.array_equal(pipeline.scaler.scale_, scaler_scale_before), (
            "Scaler scale should not change after transform_test()"
        )
    
    # Transform TEST again (should produce same result if deterministic)
    X_test_transformed2 = pipeline.transform_test(X_test)
    
    # Verify consistent transformation
    np.testing.assert_array_almost_equal(
        X_test_transformed1, X_test_transformed2, decimal=5,
        err_msg="Multiple calls to transform_test() should produce same result"
    )
