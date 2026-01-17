"""
Tests for model_utils.py - fresh_model() function
"""
import sys
from pathlib import Path
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.core.model_utils import fresh_model


def test_fresh_model_sklearn_lr():
    """Test fresh_model with sklearn LogisticRegression"""
    original = LogisticRegression(random_state=42, max_iter=100)
    fresh = fresh_model(original)
    
    assert fresh is not original
    assert isinstance(fresh, LogisticRegression)
    assert fresh.random_state == 42
    assert fresh.max_iter == 100
    assert not hasattr(fresh, 'classes_')  # Should be unfitted


def test_fresh_model_sklearn_dt():
    """Test fresh_model with sklearn DecisionTree"""
    original = DecisionTreeClassifier(random_state=42, max_depth=5)
    fresh = fresh_model(original)
    
    assert fresh is not original
    assert isinstance(fresh, DecisionTreeClassifier)
    assert fresh.random_state == 42
    assert fresh.max_depth == 5
    assert not hasattr(fresh, 'tree_')  # Should be unfitted


def test_fresh_model_sklearn_rf():
    """Test fresh_model with sklearn RandomForest"""
    original = RandomForestClassifier(random_state=42, n_estimators=50)
    fresh = fresh_model(original)
    
    assert fresh is not original
    assert isinstance(fresh, RandomForestClassifier)
    assert fresh.random_state == 42
    assert fresh.n_estimators == 50
    assert not hasattr(fresh, 'estimators_')  # Should be unfitted


def test_fresh_model_preserves_parameters():
    """Test that fresh_model preserves all parameters"""
    original = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced',
        C=0.1
    )
    fresh = fresh_model(original)
    
    assert fresh.random_state == 42
    assert fresh.max_iter == 1000
    assert fresh.class_weight == 'balanced'
    assert fresh.C == 0.1


def test_fresh_model_unfitted_after_fitting():
    """Test that fresh_model creates unfitted model even from fitted one"""
    # Fit original model
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    original = LogisticRegression(random_state=42)
    original.fit(X, y)
    
    assert hasattr(original, 'classes_')
    
    # Create fresh model
    fresh = fresh_model(original)
    
    assert fresh is not original
    assert not hasattr(fresh, 'classes_')  # Should be unfitted
    assert isinstance(fresh, LogisticRegression)


def test_fresh_model_cnn():
    """Test fresh_model with CNN model (if available)"""
    try:
        from src.models.cnn import CNNTabularClassifier
        
        original = CNNTabularClassifier(
            hidden_dims=[64, 32],
            learning_rate=0.001,
            batch_size=32,
            random_state=42
        )
        fresh = fresh_model(original)
        
        assert fresh is not original
        assert isinstance(fresh, CNNTabularClassifier)
        assert fresh.learning_rate == 0.001
        assert fresh.batch_size == 32
        assert fresh.random_state == 42
        assert fresh.hidden_dims == [64, 32]
        # Model should not be initialized yet
        assert fresh.model is None or isinstance(fresh.model, type(None))
    except ImportError:
        pytest.skip("CNN not available")


def test_fresh_model_tabnet():
    """Test fresh_model with TabNet model (if available)"""
    try:
        from src.models.tabnet import TabNetClassifierWrapper
        
        original = TabNetClassifierWrapper(
            n_d=8,
            n_a=8,
            seed=42,
            max_epochs=50
        )
        fresh = fresh_model(original)
        
        assert fresh is not original
        assert isinstance(fresh, TabNetClassifierWrapper)
        assert fresh.n_d == 8
        assert fresh.n_a == 8
        assert fresh.seed == 42
        assert fresh.max_epochs == 50
        # Model should not be initialized yet
        assert fresh.model is None
    except ImportError:
        pytest.skip("TabNet not available")


def test_fresh_model_independence():
    """Test that fresh model is independent - fitting one doesn't affect the other"""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    original = LogisticRegression(random_state=42)
    fresh = fresh_model(original)
    
    # Fit fresh model
    fresh.fit(X, y)
    
    # Original should still be unfitted
    assert not hasattr(original, 'classes_')
    
    # Fresh should be fitted
    assert hasattr(fresh, 'classes_')
