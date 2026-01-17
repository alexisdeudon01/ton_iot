import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.evaluation_3d import Evaluation3D, ResourceMonitor, ExplainabilityEvaluator
from src.models.cnn import CNNTabularClassifier, TORCH_AVAILABLE
from src.models.tabnet import TabNetClassifierWrapper, TABNET_AVAILABLE
from sklearn.linear_model import LogisticRegression

@pytest.fixture
def dummy_data():
    X = np.random.rand(200, 10)
    y = np.random.randint(0, 2, 200)
    feature_names = [f"feat_{i}" for i in range(10)]
    return X, y, feature_names

def test_resource_monitor():
    monitor = ResourceMonitor()
    monitor.start()
    # Simulate some work
    _ = [i**2 for i in range(10000)]
    monitor.update()
    metrics = monitor.stop()

    assert 'training_time_seconds' in metrics
    assert 'memory_used_mb' in metrics
    assert metrics['training_time_seconds'] > 0

def test_performance_metrics(dummy_data):
    X, y, feature_names = dummy_data
    evaluator = Evaluation3D(feature_names=feature_names)

    model = LogisticRegression()
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]

    results = evaluator.evaluate_model(
        model, "LogisticRegression",
        X_train, y_train, X_test, y_test,
        compute_explainability=False
    )

    assert 'f1_score' in results
    assert 'accuracy' in results
    assert 'precision' in results
    assert 'recall' in results
    assert 0 <= results['f1_score'] <= 1

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
def test_cnn_evaluation(dummy_data):
    X, y, feature_names = dummy_data
    evaluator = Evaluation3D(feature_names=feature_names)

    model = CNNTabularClassifier(epochs=2, batch_size=16)
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]

    results = evaluator.evaluate_model(
        model, "CNN",
        X_train, y_train, X_test, y_test,
        compute_explainability=False
    )

    assert results['model_name'] == "CNN"
    assert 'f1_score' in results

@pytest.mark.skipif(not TABNET_AVAILABLE, reason="TabNet not available")
def test_tabnet_evaluation(dummy_data):
    X, y, feature_names = dummy_data
    evaluator = Evaluation3D(feature_names=feature_names)

    model = TabNetClassifierWrapper(max_epochs=2, batch_size=16)
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]

    results = evaluator.evaluate_model(
        model, "TabNet",
        X_train, y_train, X_test, y_test,
        compute_explainability=False
    )

    assert results['model_name'] == "TabNet"
    assert 'f1_score' in results

def test_dimension_scores(dummy_data):
    X, y, feature_names = dummy_data
    evaluator = Evaluation3D(feature_names=feature_names)

    model = LogisticRegression()
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]

    evaluator.evaluate_model(model, "Model1", X_train, y_train, X_test, y_test, compute_explainability=False)
    evaluator.evaluate_model(model, "Model2", X_train, y_train, X_test, y_test, compute_explainability=False)

    scores = evaluator.get_dimension_scores()
    assert 'detection_performance' in scores.columns
    assert 'resource_efficiency' in scores.columns
    assert 'explainability' in scores.columns
    assert len(scores) == 2
