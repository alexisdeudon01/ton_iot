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
    
    assert clf.n_d == 8
    assert clf.n_a == 8
    assert clf.n_steps == 3
    assert clf.seed == 42
    assert clf.max_epochs == 100
    assert clf.model is None
    assert clf.input_dim is None


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
    
    assert clf.n_d == 16
    assert clf.n_a == 16
    assert clf.n_steps == 5
    assert clf.gamma == 2.0
    assert clf.seed == 123
    assert clf.max_epochs == 50


def test_tabnet_classifier_fit_predict():
    """Test TabNetClassifierWrapper fit and predict"""
    np.random.seed(42)
    n_samples = 300
    n_features = 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples)
    
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
    
    assert clf.model is not None
    assert clf.input_dim == n_features
    assert hasattr(clf.label_encoder, 'classes_')
    
    # Predict
    y_pred = clf.predict(X[:20])
    
    assert len(y_pred) == 20
    assert all(pred in clf.label_encoder.classes_ for pred in y_pred)


def test_tabnet_classifier_predict_proba():
    """Test TabNetClassifierWrapper predict_proba"""
    np.random.seed(42)
    n_samples = 300
    n_features = 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples)
    
    clf = TabNetClassifierWrapper(
        max_epochs=5,
        verbose=0,
        seed=42
    )
    
    clf.fit(X, y)
    
    proba = clf.predict_proba(X[:20])
    
    assert proba.shape == (20, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_tabnet_classifier_multiclass():
    """Test TabNetClassifierWrapper with multiclass classification"""
    np.random.seed(42)
    n_samples = 400
    n_features = 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 3, n_samples)  # 3 classes
    
    clf = TabNetClassifierWrapper(
        max_epochs=5,
        verbose=0,
        seed=42
    )
    
    clf.fit(X, y)
    
    y_pred = clf.predict(X[:20])
    proba = clf.predict_proba(X[:20])
    
    assert len(np.unique(y)) == 3
    assert proba.shape == (20, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_tabnet_classifier_sklearn_interface():
    """Test TabNetClassifierWrapper follows sklearn interface"""
    clf = TabNetClassifierWrapper(max_epochs=1, verbose=0)
    
    # Basic sklearn interface checks
    assert hasattr(clf, 'fit')
    assert hasattr(clf, 'predict')
    assert hasattr(clf, 'predict_proba')
    assert hasattr(clf, 'get_params')
    assert hasattr(clf, 'set_params')


def test_tabnet_classifier_parameters_preserved():
    """Test that fit preserves all parameters"""
    clf = TabNetClassifierWrapper(
        n_d=16,
        n_a=16,
        gamma=2.0,
        seed=99,
        max_epochs=10
    )
    
    np.random.seed(42)
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randint(0, 2, 100)
    
    clf.fit(X, y)
    
    # Parameters should be preserved after fit
    assert clf.n_d == 16
    assert clf.n_a == 16
    assert clf.gamma == 2.0
    assert clf.seed == 99
