"""
Comprehensive tests for Evaluation3D framework
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.evaluation_3d import Evaluation3D, ResourceMonitor, ExplainabilityEvaluator
from src.models.cnn import CNNTabularClassifier, TORCH_AVAILABLE
from src.models.tabnet import TabNetClassifierWrapper, TABNET_AVAILABLE
from sklearn.linear_model import LogisticRegression

@pytest.fixture
def dummy_data():
    """Fixture for synthetic evaluation data"""
    np.random.seed(42)
    X = np.random.rand(200, 10).astype(np.float32)
    y = np.random.randint(0, 2, 200)
    feature_names = [f"feat_{i}" for i in range(10)]
    return X, y, feature_names

def test_resource_monitor():
    """Test ResourceMonitor measures time and memory correctly"""
    monitor = ResourceMonitor()
    monitor.start()
    # Simulate some work
    _ = [i**2 for i in range(10000)]
    monitor.update()
    metrics = monitor.stop()

    assert 'training_time_seconds' in metrics, \
        f"Metrics should contain 'training_time_seconds' (got keys: {list(metrics.keys())})"
    assert 'memory_used_mb' in metrics, \
        f"Metrics should contain 'memory_used_mb' (got keys: {list(metrics.keys())})"
    assert metrics['training_time_seconds'] > 0, \
        f"Training time should be positive (got {metrics['training_time_seconds']})"
    assert metrics['memory_used_mb'] >= 0, \
        f"Memory usage should be non-negative (got {metrics['memory_used_mb']})"

def test_performance_metrics(dummy_data):
    """Test performance metrics computation"""
    X, y, feature_names = dummy_data
    evaluator = Evaluation3D(feature_names=feature_names)

    model = LogisticRegression(random_state=42, max_iter=1000)
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]

    results = evaluator.evaluate_model(
        model, "LogisticRegression",
        X_train, y_train, X_test, y_test,
        compute_explainability=False
    )

    assert 'f1_score' in results, \
        f"Results should contain 'f1_score' (got keys: {list(results.keys())})"
    assert 'accuracy' in results, \
        f"Results should contain 'accuracy' (got keys: {list(results.keys())})"
    assert 'precision' in results, \
        f"Results should contain 'precision' (got keys: {list(results.keys())})"
    assert 'recall' in results, \
        f"Results should contain 'recall' (got keys: {list(results.keys())})"
    assert 0 <= results['f1_score'] <= 1, \
        f"F1 score should be in [0, 1] (got {results['f1_score']})"
    assert 0 <= results['accuracy'] <= 1, \
        f"Accuracy should be in [0, 1] (got {results['accuracy']})"

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
def test_cnn_evaluation(dummy_data):
    """Test CNN model evaluation"""
    X, y, feature_names = dummy_data
    evaluator = Evaluation3D(feature_names=feature_names)

    model = CNNTabularClassifier(epochs=2, batch_size=16, random_state=42)
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]

    results = evaluator.evaluate_model(
        model, "CNN",
        X_train, y_train, X_test, y_test,
        compute_explainability=False
    )

    assert results['model_name'] == "CNN", \
        f"Model name should be 'CNN' (got {results.get('model_name')})"
    assert 'f1_score' in results, \
        f"Results should contain 'f1_score' (got keys: {list(results.keys())})"
    assert 0 <= results['f1_score'] <= 1, \
        f"F1 score should be in [0, 1] (got {results['f1_score']})"

@pytest.mark.skipif(not TABNET_AVAILABLE, reason="TabNet not available")
def test_tabnet_evaluation(dummy_data):
    """Test TabNet model evaluation"""
    X, y, feature_names = dummy_data
    evaluator = Evaluation3D(feature_names=feature_names)

    model = TabNetClassifierWrapper(max_epochs=2, batch_size=16, verbose=0, seed=42)
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]

    results = evaluator.evaluate_model(
        model, "TabNet",
        X_train, y_train, X_test, y_test,
        compute_explainability=False
    )

    assert results['model_name'] == "TabNet", \
        f"Model name should be 'TabNet' (got {results.get('model_name')})"
    assert 'f1_score' in results, \
        f"Results should contain 'f1_score' (got keys: {list(results.keys())})"
    assert 0 <= results['f1_score'] <= 1, \
        f"F1 score should be in [0, 1] (got {results['f1_score']})"

def test_dimension_scores(dummy_data):
    """Test dimension scores aggregation across multiple models"""
    X, y, feature_names = dummy_data
    evaluator = Evaluation3D(feature_names=feature_names)

    model = LogisticRegression(random_state=42, max_iter=1000)
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]

    evaluator.evaluate_model(model, "Model1", X_train, y_train, X_test, y_test, compute_explainability=False)
    evaluator.evaluate_model(model, "Model2", X_train, y_train, X_test, y_test, compute_explainability=False)

    scores = evaluator.get_dimension_scores()
    assert 'detection_performance' in scores.columns, \
        f"Scores should contain 'detection_performance' (got columns: {list(scores.columns)})"
    assert 'resource_efficiency' in scores.columns, \
        f"Scores should contain 'resource_efficiency' (got columns: {list(scores.columns)})"
    assert 'explainability' in scores.columns, \
        f"Scores should contain 'explainability' (got columns: {list(scores.columns)})"
    assert len(scores) == 2, f"Should have scores for 2 models (got {len(scores)})"
    assert all(scores['detection_performance'].between(0, 1)), \
        f"All detection_performance scores should be in [0, 1] (got {scores['detection_performance'].tolist()})"
