"""
Tests for model-aware preprocessing profiles
Verifies that LR/Tree/CNN/TabNet profiles are applied correctly with different preprocessing steps
"""
import sys
from pathlib import Path
import pytest

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import PipelineConfig
from src.phases.phase3_evaluation import Phase3Evaluation


def test_model_aware_profiles():
    """
    Test that preprocessing profiles are model-aware.
    
    Input:
        - PipelineConfig with preprocessing_profiles
        - Different model names (LR, DT, RF, CNN, TabNet)
    
    Processing:
        - Get preprocessing profile for each model
        - Verify profile settings match expected model requirements
    
    Expected Output:
        - LR profile: scaling=True, feature_selection=True, resampling=True
        - Tree profile (DT/RF): scaling=False, feature_selection=False, resampling=False, class_weight=True
        - CNN profile: scaling=True, feature_selection=False, resampling=True, cnn_reshape=True
        - TabNet profile: scaling=False, feature_selection=False, resampling=False, class_weight=True
    
    Method:
        - Direct testing of Phase3Evaluation._get_preprocessing_profile()
    """
    config = PipelineConfig()
    phase3 = Phase3Evaluation(config)

    # Test LR profile
    lr_profile = phase3._get_preprocessing_profile("LR", n_features=50)
    assert lr_profile["apply_scaling"] is True, \
        "Logistic Regression should use scaling (linear models benefit from scaling)"
    assert lr_profile["apply_feature_selection"] is True, \
        "Logistic Regression should use feature selection"
    assert lr_profile["apply_resampling"] is True, \
        "Logistic Regression should use resampling"
    assert lr_profile.get("use_class_weight", False) is False, \
        "Logistic Regression should not use class_weight by default"
    # Verify feature_selection_k is calculated dynamically
    assert "feature_selection_k" in lr_profile, \
        "feature_selection_k should be calculated when feature_selection_k_dynamic=True"
    assert 10 <= lr_profile["feature_selection_k"] <= 60, \
        f"feature_selection_k should be in [10, 60], got {lr_profile['feature_selection_k']}"

    # Test Tree profile (DT and RF)
    dt_profile = phase3._get_preprocessing_profile("DT", n_features=50)
    rf_profile = phase3._get_preprocessing_profile("RF", n_features=50)
    
    for tree_profile in [dt_profile, rf_profile]:
        assert tree_profile["apply_scaling"] is False, \
            "Tree-based models should not use scaling (scale-invariant)"
        assert tree_profile["apply_feature_selection"] is False, \
            "Tree-based models should not use feature selection (handle feature importance internally)"
        assert tree_profile["apply_resampling"] is False, \
            "Tree-based models should not use resampling (use class_weight instead)"
        assert tree_profile["use_class_weight"] is True, \
            "Tree-based models should use class_weight for imbalanced data"
        assert tree_profile.get("class_weight") == "balanced", \
            "Tree-based models should use class_weight='balanced'"

    # Test CNN profile
    cnn_profile = phase3._get_preprocessing_profile("CNN", n_features=50)
    assert cnn_profile["apply_scaling"] is True, \
        "CNN should use scaling (neural networks require normalized inputs)"
    assert cnn_profile["apply_feature_selection"] is False, \
        "CNN should not use feature selection (learns features automatically)"
    assert cnn_profile["apply_resampling"] is True, \
        "CNN should use resampling for imbalanced data"
    assert cnn_profile.get("cnn_reshape", False) is True, \
        "CNN profile should have cnn_reshape=True for proper input shape"

    # Test TabNet profile
    tabnet_profile = phase3._get_preprocessing_profile("TabNet", n_features=50)
    assert tabnet_profile["apply_scaling"] is False, \
        "TabNet handles scaling internally (pytorch-tabnet)"
    assert tabnet_profile["apply_feature_selection"] is False, \
        "TabNet should not use external feature selection"
    assert tabnet_profile["apply_resampling"] is False, \
        "TabNet uses class_weight instead of resampling"
    assert tabnet_profile["use_class_weight"] is True, \
        "TabNet should use class_weight for imbalanced data"
    assert tabnet_profile.get("class_weight") == "balanced", \
        "TabNet should use class_weight='balanced'"