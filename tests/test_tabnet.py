"""
Tests for tabnet.py - TabNetClassifierWrapper
"""
import sys
from pathlib import Path
import pytest
import numpy as np

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

pytest.importorskip("pytorch_tabnet", reason="pytorch-tabnet not available")

from src.models.tabnet import TabNetClassifierWrapper


def test_tabnet_classifier_init_default():
    """Test TabNetClassifierWrapper initialization with defaults"""
    clf = TabNetClassifierWrapper()
    
    assert clf.n_d == 8, f"n_d should be 8 (got {clf.n_d})"
    assert clf.n_a == 8, f"n_a should be 8 (got {clf.n_a})"
    assert clf.n_steps == 3, f"n_steps should be 3 (got {clf.n_steps})"
    assert clf.seed == 42, f"seed should be 42 (got {clf.seed})"
    assert clf.max_epochs == 100, f"max_epochs should be 100 (got {clf.max_epochs})"
    assert clf.model is None, "Model should not be initialized before fit()"
    assert clf.input_dim is None, "input_dim should be None before fit()"


def test_tabnet_classifier_init_custom():
    """Test TabNetClassifierWrapper initialization with custom parameters"""
    clf = TabNetClassifierWrapper(
        n_d=16,
        n_a=16,
        n_steps=5,
        gamma=2.0,
        seed=123,
        max_epochs=50
    )
    
    assert clf.n_d == 16, f"n_d should be 16 (got {clf.n_d})"
    assert clf.n_a == 16, f"n_a should be 16 (got {clf.n_a})"
    assert clf.n_steps == 5, f"n_steps should be 5 (got {clf.n_steps})"
    assert clf.gamma == 2.0, f"gamma should be 2.0 (got {clf.gamma})"
    assert clf.seed == 123, f"seed should be 123 (got {clf.seed})"
    assert clf.max_epochs == 50, f"max_epochs should be 50 (got {clf.max_epochs})"


def test_tabnet_classifier_fit_predict(synthetic_binary_data):
    """Test TabNetClassifierWrapper fit and predict"""
    X, y = synthetic_binary_data
    n_features = X.shape[1]
    
    clf = TabNetClassifierWrapper(
        n_d=8,
        n_a=8,
        max_epochs=5,  # Short training for test
        batch_size=128,
        verbose=0,
        seed=42
    )
    
    # Fit
    clf.fit(X, y)
    
    assert clf.model is not None, "Model should be created after fit()"
    assert clf.input_dim == n_features, f"input_dim should be {n_features} (got {clf.input_dim})"
    assert hasattr(clf.label_encoder, 'classes_'), "Label encoder should have classes_ after fit()"
    
    # Predict
    y_pred = clf.predict(X[:20])
    
    assert len(y_pred) == 20, f"Prediction length should be 20 (got {len(y_pred)})"
    assert all(pred in clf.label_encoder.classes_ for pred in y_pred), "All predictions should be in label_encoder.classes_"


def test_tabnet_classifier_predict_proba(synthetic_binary_data):
    """Test TabNetClassifierWrapper predict_proba"""
    X, y = synthetic_binary_data
    
    clf = TabNetClassifierWrapper(
        max_epochs=5,
        verbose=0,
        seed=42
    )
    
    clf.fit(X, y)
    
    proba = clf.predict_proba(X[:20])
    
    assert proba.shape == (20, 2), f"Probability shape should be (20, 2) (got {proba.shape})"
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5), "Probabilities should sum to 1.0 per sample"
    assert (proba >= 0).all() and (proba <= 1).all(), "All probabilities should be in [0, 1] range"


def test_tabnet_classifier_multiclass(synthetic_multiclass_data):
    """Test TabNetClassifierWrapper with multiclass classification"""
    X, y = synthetic_multiclass_data
    
    clf = TabNetClassifierWrapper(
        max_epochs=5,
        verbose=0,
        seed=42
    )
    
    clf.fit(X, y)
    
    y_pred = clf.predict(X[:20])
    proba = clf.predict_proba(X[:20])
    
    assert len(np.unique(y)) == 3, f"Should have 3 unique classes (got {len(np.unique(y))})"
    assert proba.shape == (20, 3), f"Probability shape should be (20, 3) (got {proba.shape})"
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5), "Probabilities should sum to 1.0 per sample"


def test_tabnet_classifier_sklearn_interface():
    """Test TabNetClassifierWrapper follows sklearn interface"""
    clf = TabNetClassifierWrapper(max_epochs=1, verbose=0)
    
    # Basic sklearn interface checks
    assert hasattr(clf, 'fit'), "Classifier should have fit() method"
    assert hasattr(clf, 'predict'), "Classifier should have predict() method"
    assert hasattr(clf, 'predict_proba'), "Classifier should have predict_proba() method"
    assert hasattr(clf, 'get_params'), "Classifier should have get_params() method"
    assert hasattr(clf, 'set_params'), "Classifier should have set_params() method"


def test_tabnet_classifier_parameters_preserved(synthetic_small_data):
    """Test that fit preserves all parameters"""
    X, y = synthetic_small_data
    
    clf = TabNetClassifierWrapper(
        n_d=16,
        n_a=16,
        gamma=2.0,
        seed=99,
        max_epochs=10
    )
    
    clf.fit(X, y)
    
    # Parameters should be preserved after fit
    assert clf.n_d == 16, f"n_d should be preserved as 16 (got {clf.n_d})"
    assert clf.n_a == 16, f"n_a should be preserved as 16 (got {clf.n_a})"
    assert clf.gamma == 2.0, f"gamma should be preserved as 2.0 (got {clf.gamma})"
    assert clf.seed == 99, f"seed should be preserved as 99 (got {clf.seed})"
