"""
Tests for model-aware preprocessing profiles
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
    """Test that preprocessing profiles are model-aware"""
    config = PipelineConfig()
    phase3 = Phase3Evaluation(config)

    lr_profile = phase3._get_preprocessing_profile("Logistic Regression", n_features=50)
    tree_profile = phase3._get_preprocessing_profile("Random Forest", n_features=50)
    nn_profile = phase3._get_preprocessing_profile("CNN", n_features=50)

    # Logistic Regression should use scaling, feature selection, and resampling
    assert lr_profile["apply_scaling"] is True, \
        "Logistic Regression should use scaling (linear models benefit from scaling)"
    assert lr_profile["apply_feature_selection"] is True, \
        "Logistic Regression should use feature selection"
    assert lr_profile["apply_resampling"] is True, \
        "Logistic Regression should use resampling"

    # Tree-based models should not use scaling/feature selection, but use class_weight
    assert tree_profile["apply_scaling"] is False, \
        "Random Forest should not use scaling (tree-based models are scale-invariant)"
    assert tree_profile["apply_feature_selection"] is False, \
        "Random Forest should not use feature selection (trees handle feature importance internally)"
    assert tree_profile["apply_resampling"] is False, \
        "Random Forest should not use resampling (uses class_weight instead)"
    assert tree_profile["use_class_weight"] is True, \
        "Random Forest should use class_weight for imbalanced data"

    # Neural networks should use scaling and resampling, but not feature selection
    assert nn_profile["apply_scaling"] is True, \
        "CNN should use scaling (neural networks require normalized inputs)"
    assert nn_profile["apply_feature_selection"] is False, \
        "CNN should not use feature selection (neural networks learn features automatically)"
    assert nn_profile["apply_resampling"] is True, \
        "CNN should use resampling"
