from src.config import PipelineConfig
from src.phases.phase3_evaluation import Phase3Evaluation


def test_model_aware_profiles():
    config = PipelineConfig()
    phase3 = Phase3Evaluation(config)

    lr_profile = phase3._get_preprocessing_profile("Logistic Regression", n_features=50)
    tree_profile = phase3._get_preprocessing_profile("Random Forest", n_features=50)
    nn_profile = phase3._get_preprocessing_profile("CNN", n_features=50)

    assert lr_profile["apply_scaling"] is True
    assert lr_profile["apply_feature_selection"] is True
    assert lr_profile["apply_resampling"] is True

    assert tree_profile["apply_scaling"] is False
    assert tree_profile["apply_feature_selection"] is False
    assert tree_profile["apply_resampling"] is False
    assert tree_profile["use_class_weight"] is True

    assert nn_profile["apply_scaling"] is True
    assert nn_profile["apply_feature_selection"] is False
    assert nn_profile["apply_resampling"] is True
