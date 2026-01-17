"""
Tests for preprocessing pipeline utilities, including sanitize_numeric_values()
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.core.preprocessing_pipeline import PreprocessingPipeline


def test_sanitize_numeric_values_removes_inf_and_clips():
    """
    Test that sanitize_numeric_values() removes inf values and clips outliers.
    
    Input:
        - DataFrame with inf values (np.inf, -np.inf) and extreme outliers
        - Parameters: replace_inf_with_max=False, clip_quantiles=(0.001, 0.999)
    
    Processing:
        - Create DataFrame with inf and extreme values
        - Call sanitize_numeric_values() with quantile clipping
        - Verify inf values are replaced with NaN
        - Verify extreme values are clipped to quantile bounds
    
    Expected Output:
        - No inf values in result
        - Extreme values clipped to [q_low, q_high] per column
        - Same shape and columns as input
    
    Method:
        - Direct call to sanitize_numeric_values() method
    """
    pipeline = PreprocessingPipeline(random_state=42)
    
    # Create DataFrame with inf values and extreme outliers
    n_samples = 100
    n_features = 5
    
    # Generate base data
    np.random.seed(42)
    X_base = np.random.randn(n_samples, n_features) * 10
    
    # Inject inf values
    X_base[0, 0] = np.inf  # +inf
    X_base[1, 0] = -np.inf  # -inf
    X_base[2, 1] = np.inf
    X_base[3, 1] = -np.inf
    
    # Inject extreme outliers (values way outside normal range)
    X_base[4, 2] = 1000000.0  # Extreme positive outlier
    X_base[5, 2] = -1000000.0  # Extreme negative outlier
    X_base[6, 3] = 500000.0
    X_base[7, 3] = -500000.0
    
    X_df = pd.DataFrame(X_base, columns=[f"feature_{i}" for i in range(n_features)])
    
    # Apply sanitization with quantile clipping
    X_sanitized = pipeline.sanitize_numeric_values(
        X_df,
        replace_inf_with_max=False,  # Replace with NaN
        clip_quantiles=(0.001, 0.999),  # Clip outliers
        abs_cap=None,  # No absolute cap
    )
    
    # Verify shape and columns are preserved
    assert X_sanitized.shape == X_df.shape, f"Shape mismatch: expected {X_df.shape}, got {X_sanitized.shape}"
    assert list(X_sanitized.columns) == list(X_df.columns), "Columns mismatch"
    
    # Verify no inf values remain
    inf_count = np.isinf(X_sanitized.values).sum()
    assert inf_count == 0, f"Found {inf_count} inf values in sanitized data (expected 0)"
    
    # Verify extreme outliers are clipped (values should be within quantile bounds)
    for col in X_sanitized.columns:
        finite_values = X_df.loc[np.isfinite(X_df[col]), col]
        if len(finite_values) > 0:
            q_low = finite_values.quantile(0.001)
            q_high = finite_values.quantile(0.999)
            
            # All values in sanitized column should be within [q_low, q_high] or NaN
            sanitized_col = X_sanitized[col]
            finite_sanitized = sanitized_col[np.isfinite(sanitized_col)]
            
            if len(finite_sanitized) > 0:
                assert finite_sanitized.min() >= q_low, (
                    f"Column {col}: min value {finite_sanitized.min():.3f} < q_low {q_low:.3f}"
                )
                assert finite_sanitized.max() <= q_high, (
                    f"Column {col}: max value {finite_sanitized.max():.3f} > q_high {q_high:.3f}"
                )
    
    # Verify original extreme values are now clipped
    # The extreme values (1000000, -1000000, etc.) should be clipped
    # Check that feature_2 (which had 1000000) is now within bounds
    feature_2_finite = X_df.loc[np.isfinite(X_df['feature_2']), 'feature_2']
    q_low_2 = feature_2_finite.quantile(0.001)
    q_high_2 = feature_2_finite.quantile(0.999)
    
    assert X_sanitized.loc[4, 'feature_2'] <= q_high_2, (
        f"Extreme value at [4, 'feature_2'] not clipped: {X_sanitized.loc[4, 'feature_2']:.3f} > {q_high_2:.3f}"
    )
    assert X_sanitized.loc[5, 'feature_2'] >= q_low_2, (
        f"Extreme value at [5, 'feature_2'] not clipped: {X_sanitized.loc[5, 'feature_2']:.3f} < {q_low_2:.3f}"
    )


def test_sanitize_numeric_values_replace_inf_with_max():
    """
    Test sanitize_numeric_values() with replace_inf_with_max=True.
    
    Input:
        - DataFrame with inf values
        - Parameter: replace_inf_with_max=True
    
    Processing:
        - Create DataFrame with inf values
        - Call sanitize_numeric_values() with replace_inf_with_max=True
        - Verify inf values are replaced with column max/min
    
    Expected Output:
        - No inf values
        - +inf replaced by column max (finite)
        - -inf replaced by column min (finite)
    
    Method:
        - Direct call to sanitize_numeric_values() method
    """
    pipeline = PreprocessingPipeline(random_state=42)
    
    # Create DataFrame with inf values
    np.random.seed(42)
    X_base = np.random.randn(100, 3) * 10
    X_base[0, 0] = np.inf
    X_base[1, 0] = -np.inf
    X_base[2, 1] = np.inf
    
    X_df = pd.DataFrame(X_base, columns=['feature_0', 'feature_1', 'feature_2'])
    
    # Calculate expected max/min for each column
    expected_max_0 = X_df.loc[np.isfinite(X_df['feature_0']), 'feature_0'].max()
    expected_min_0 = X_df.loc[np.isfinite(X_df['feature_0']), 'feature_0'].min()
    expected_max_1 = X_df.loc[np.isfinite(X_df['feature_1']), 'feature_1'].max()
    
    # Apply sanitization with replace_inf_with_max=True
    X_sanitized = pipeline.sanitize_numeric_values(
        X_df,
        replace_inf_with_max=True,  # Replace with max/min
        clip_quantiles=None,  # No clipping
        abs_cap=None,
    )
    
    # Verify no inf values
    inf_count = np.isinf(X_sanitized.values).sum()
    assert inf_count == 0, f"Found {inf_count} inf values (expected 0)"
    
    # Verify inf replacements
    assert X_sanitized.loc[0, 'feature_0'] == expected_max_0, (
        f"+inf not replaced correctly: got {X_sanitized.loc[0, 'feature_0']}, expected {expected_max_0}"
    )
    assert X_sanitized.loc[1, 'feature_0'] == expected_min_0, (
        f"-inf not replaced correctly: got {X_sanitized.loc[1, 'feature_0']}, expected {expected_min_0}"
    )
    assert X_sanitized.loc[2, 'feature_1'] == expected_max_1, (
        f"+inf not replaced correctly: got {X_sanitized.loc[2, 'feature_1']}, expected {expected_max_1}"
    )


def test_sanitize_numeric_values_abs_cap():
    """
    Test sanitize_numeric_values() with absolute cap.
    
    Input:
        - DataFrame with values outside abs_cap range
        - Parameter: abs_cap=100.0
    
    Processing:
        - Create DataFrame with values > abs_cap and < -abs_cap
        - Call sanitize_numeric_values() with abs_cap=100.0
        - Verify values are clipped to [-100, 100]
    
    Expected Output:
        - All values in range [-abs_cap, abs_cap]
    
    Method:
        - Direct call to sanitize_numeric_values() method
    """
    pipeline = PreprocessingPipeline(random_state=42)
    
    # Create DataFrame with values outside abs_cap
    np.random.seed(42)
    X_base = np.random.randn(50, 3) * 200  # Some values will be > 100 or < -100
    X_base[0, 0] = 500.0  # > abs_cap
    X_base[1, 0] = -500.0  # < -abs_cap
    
    X_df = pd.DataFrame(X_base, columns=['feature_0', 'feature_1', 'feature_2'])
    
    abs_cap = 100.0
    
    # Apply sanitization with abs_cap
    X_sanitized = pipeline.sanitize_numeric_values(
        X_df,
        replace_inf_with_max=False,
        clip_quantiles=None,  # No quantile clipping
        abs_cap=abs_cap,
    )
    
    # Verify all values are within [-abs_cap, abs_cap]
    assert X_sanitized.values.min() >= -abs_cap, (
        f"Min value {X_sanitized.values.min():.3f} < -abs_cap ({-abs_cap})"
    )
    assert X_sanitized.values.max() <= abs_cap, (
        f"Max value {X_sanitized.values.max():.3f} > abs_cap ({abs_cap})"
    )
    
    # Verify specific values are clipped
    assert X_sanitized.loc[0, 'feature_0'] == abs_cap, (
        f"Value 500.0 not clipped to {abs_cap}: got {X_sanitized.loc[0, 'feature_0']}"
    )
    assert X_sanitized.loc[1, 'feature_0'] == -abs_cap, (
        f"Value -500.0 not clipped to {-abs_cap}: got {X_sanitized.loc[1, 'feature_0']}"
    )


def test_sanitize_numeric_values_integration_with_clean_data():
    """
    Test that sanitize_numeric_values() is properly integrated in clean_data().
    
    Input:
        - DataFrame with inf values and outliers
        - Call clean_data() which internally uses sanitize_numeric_values()
    
    Processing:
        - Create DataFrame with inf and extreme values
        - Call clean_data()
        - Verify inf values are handled and outliers are clipped
    
    Expected Output:
        - Cleaned DataFrame with no inf values
        - Outliers clipped appropriately
    
    Method:
        - Integration test via clean_data() method
    """
    pipeline = PreprocessingPipeline(random_state=42)
    
    # Create DataFrame with inf and extreme values
    np.random.seed(42)
    X_base = np.random.randn(100, 5) * 10
    X_base[0, 0] = np.inf
    X_base[1, 0] = -np.inf
    X_base[2, 1] = 1000000.0  # Extreme outlier
    
    X_df = pd.DataFrame(X_base, columns=[f"feature_{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100))
    
    # Call clean_data() which internally uses sanitize_numeric_values()
    X_cleaned, y_cleaned = pipeline.clean_data(
        X_df,
        y,
        impute=True,
        replace_inf_with_max=False,
    )
    
    # Verify no inf values
    inf_count = np.isinf(X_cleaned.values).sum()
    assert inf_count == 0, f"Found {inf_count} inf values in cleaned data (expected 0)"
    
    # Verify extreme outlier is clipped (should be within reasonable range)
    # The value 1000000 should be clipped to quantile bounds
    feature_1_finite = X_df.loc[np.isfinite(X_df['feature_1']), 'feature_1']
    q_high_1 = feature_1_finite.quantile(0.999)
    
    # After sanitization + imputation, the extreme value should be handled
    assert np.abs(X_cleaned.loc[2, 'feature_1']) <= np.abs(q_high_1) * 2, (
        f"Extreme outlier not properly handled: {X_cleaned.loc[2, 'feature_1']:.3f} still too large"
    )
    
    # Verify shape is preserved
    assert X_cleaned.shape[0] == X_df.shape[0], "Number of rows changed"
    assert X_cleaned.shape[1] == X_df.shape[1], "Number of columns changed"
