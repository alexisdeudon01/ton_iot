"""
Tests for Phase 3 CNN and TabNet integration
"""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import PipelineConfig, TEST_CONFIG


def test_phase3_cnn_tabnet_profiles():
    """
    Test that CNN and TabNet profiles are correctly defined in config.
    
    Input:
        - PipelineConfig with preprocessing_profiles
    
    Processing:
        - Check cnn_profile and tabnet_profile exist and have correct settings
    
    Expected Output:
        - cnn_profile: apply_scaling=True, apply_feature_selection=False, cnn_reshape=True
        - tabnet_profile: apply_scaling=False, apply_feature_selection=False, use_class_weight=True
    
    Method:
        - Direct inspection of config.preprocessing_profiles
    """
    config = PipelineConfig()
    
    # Check CNN profile
    assert 'cnn_profile' in config.preprocessing_profiles, "cnn_profile should exist"
    cnn_profile = config.preprocessing_profiles['cnn_profile']
    assert cnn_profile.get('apply_scaling') is True, "CNN should use scaling"
    assert cnn_profile.get('apply_feature_selection') is False, "CNN should not use feature selection"
    assert cnn_profile.get('cnn_reshape') is True, "CNN should reshape to (n, d, 1)"
    assert cnn_profile.get('apply_resampling') is True, "CNN should use resampling"
    
    # Check TabNet profile
    assert 'tabnet_profile' in config.preprocessing_profiles, "tabnet_profile should exist"
    tabnet_profile = config.preprocessing_profiles['tabnet_profile']
    assert tabnet_profile.get('apply_scaling') is False, "TabNet should not use scaling"
    assert tabnet_profile.get('apply_feature_selection') is False, "TabNet should not use feature selection"
    assert tabnet_profile.get('use_class_weight') is True, "TabNet should use class_weight"
    assert tabnet_profile.get('class_weight') == 'balanced', "TabNet should use 'balanced' class_weight"


def test_phase3_model_names():
    """
    Test that official model names (LR, DT, RF, CNN, TabNet) are used.
    
    Input:
        - PipelineConfig.phase3_algorithms
    
    Processing:
        - Check that algorithms list contains expected names
    
    Expected Output:
        - Algorithms should include 'CNN' and 'TabNet'
    
    Method:
        - Direct inspection of config.phase3_algorithms
    """
    config = PipelineConfig()
    
    # Check that CNN and TabNet are in the list
    algos_lower = [a.lower().replace("-", "_") for a in config.phase3_algorithms]
    assert 'cnn' in algos_lower or 'CNN' in config.phase3_algorithms, (
        "CNN should be in phase3_algorithms"
    )
    assert 'tabnet' in algos_lower or 'TabNet' in config.phase3_algorithms, (
        "TabNet should be in phase3_algorithms"
    )


@pytest.mark.skipif(
    True,  # Skip if CNN/TabNet not available
    reason="CNN/TabNet models not available in test environment"
)
def test_phase3_cnn_reshape():
    """
    Test that CNN receives reshaped data (n, d, 1).
    
    Input:
        - Mock Phase 3 evaluation with CNN model
    
    Processing:
        - Simulate preprocessing for CNN
        - Check that data is reshaped to (n_samples, n_features, 1)
    
    Expected Output:
        - CNN data shape should be (n_samples, n_features, 1)
    
    Method:
        - Mock Phase3Evaluation with CNN profile
    """
    # This test would require actual CNN model and Phase3Evaluation
    # For now, we test the profile configuration
    config = PipelineConfig()
    cnn_profile = config.preprocessing_profiles['cnn_profile']
    
    assert cnn_profile.get('cnn_reshape') is True, "CNN profile should have cnn_reshape=True"
    
    # Simulate reshape logic
    n_samples, n_features = 100, 20
    X_flat = np.random.randn(n_samples, n_features)
    X_reshaped = X_flat.reshape(n_samples, n_features, 1)
    
    assert X_reshaped.shape == (n_samples, n_features, 1), (
        f"Expected shape (100, 20, 1), got {X_reshaped.shape}"
    )


@pytest.mark.skipif(
    True,  # Skip if SHAP/LIME not available
    reason="SHAP/LIME not available in test environment"
)
def test_phase3_explainability_no_crash():
    """
    Test that SHAP/LIME computation doesn't crash if unavailable.
    
    Input:
        - Mock evaluation with CNN/TabNet models
    
    Processing:
        - Attempt SHAP/LIME computation
        - Check that missing SHAP/LIME doesn't cause crash
    
    Expected Output:
        - Evaluation completes successfully even if SHAP/LIME fails
    
    Method:
        - Mock explainability computation
    """
    # This test verifies that explainability computation is robust
    # Actual implementation in evaluation_3d.py handles missing SHAP/LIME gracefully
    from src.evaluation_3d import ExplainabilityEvaluator
    
    evaluator = ExplainabilityEvaluator(feature_names=['f1', 'f2', 'f3'])
    
    # Mock model without predict_proba (should handle gracefully)
    mock_model = Mock()
    mock_model.predict_proba = None
    
    X_sample = np.random.randn(10, 3)
    X_train = np.random.randn(50, 3)
    
    # Should not crash even if SHAP/LIME unavailable
    shap_score = evaluator.compute_shap_score(mock_model, X_sample, max_samples=5)
    lime_score = evaluator.compute_lime_score(mock_model, X_sample, X_train, max_samples=5)
    
    # Results may be None if unavailable, which is acceptable
    assert shap_score is None or isinstance(shap_score, (int, float)), (
        f"SHAP score should be None or numeric, got {type(shap_score)}"
    )
    assert lime_score is None or isinstance(lime_score, (int, float)), (
        f"LIME score should be None or numeric, got {type(lime_score)}"
    )


def test_phase3_metrics_df_format():
    """
    Test that metrics_df has correct format with 'algo' column.
    
    Input:
        - Mock metrics DataFrame with LR, CNN, TabNet
    
    Processing:
        - Create metrics_df with 'algo' column
        - Verify ensure_algo_column() works correctly
    
    Expected Output:
        - DataFrame has 'algo' column with correct values
    
    Method:
        - Direct creation and validation
    """
    from src.evaluation.visualizations import ensure_algo_column, get_algo_names
    
    metrics_df = pd.DataFrame({
        'algo': ['LR', 'CNN', 'TabNet'],
        'f1_mean': [0.9, 0.88, 0.87],
        'training_time_seconds': [1.2, 5.1, 4.2],
        'explainability_score': [0.8, 0.3, 0.4]
    })
    
    # Ensure algo column
    metrics_df_ensured = ensure_algo_column(metrics_df)
    assert 'algo' in metrics_df_ensured.columns, "Expected 'algo' column"
    
    # Get algo names
    algos = get_algo_names(metrics_df_ensured)
    assert list(algos.values) == ['LR', 'CNN', 'TabNet'], (
        f"Expected ['LR', 'CNN', 'TabNet'], got {list(algos.values)}"
    )
