#!/usr/bin/env python3
"""Tests for model-aware preprocessing profiles in Phase 3."""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.config import PipelineConfig
from src.phases.phase3_evaluation import Phase3Evaluation


def _make_training_data(n_samples: int = 40, n_features: int = 6):
    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        rng.normal(size=(n_samples, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series([0, 1] * (n_samples // 2))
    return X, y


def test_model_profiles_apply_expected_steps():
    """Ensure profiles drive scaling/selection/SMOTE and class_weight defaults."""
    config = PipelineConfig()
    phase3 = Phase3Evaluation(config)
    X_train, y_train = _make_training_data()

    lr_profile = phase3._get_preprocessing_profile("Logistic Regression", X_train.shape[1])
    _, _, lr_pipeline = phase3._apply_preprocessing_per_fold(X_train, y_train, lr_profile)
    assert lr_pipeline.is_fitted is True
    assert lr_pipeline.feature_selector is not None
    assert lr_pipeline.smote is not None

    tree_profile = phase3._get_preprocessing_profile("Decision Tree", X_train.shape[1])
    _, _, tree_pipeline = phase3._apply_preprocessing_per_fold(X_train, y_train, tree_profile)
    assert tree_pipeline.is_fitted is False
    assert tree_pipeline.feature_selector is None
    assert tree_pipeline.smote is None

    nn_profile = phase3._get_preprocessing_profile("CNN", X_train.shape[1])
    _, _, nn_pipeline = phase3._apply_preprocessing_per_fold(X_train, y_train, nn_profile)
    assert nn_pipeline.is_fitted is True
    assert nn_pipeline.feature_selector is None
    assert nn_pipeline.smote is not None

    model = DecisionTreeClassifier(random_state=config.random_state)
    if tree_profile.get("use_class_weight", False):
        class_weight = tree_profile.get("class_weight", "balanced")
        model.set_params(class_weight=class_weight)
    assert model.get_params()["class_weight"] == tree_profile["class_weight"]
