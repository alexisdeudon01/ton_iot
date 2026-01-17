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
    
    assert fresh is not original, "Fresh model should be a different instance"
    assert isinstance(fresh, LogisticRegression), "Fresh model should be LogisticRegression type"
    assert fresh.random_state == 42, f"random_state should be preserved (got {fresh.random_state})"
    assert fresh.max_iter == 100, f"max_iter should be preserved (got {fresh.max_iter})"
    assert not hasattr(fresh, 'classes_'), "Fresh model should be unfitted (no classes_ attribute)"


def test_fresh_model_sklearn_dt():
    """Test fresh_model with sklearn DecisionTree"""
    original = DecisionTreeClassifier(random_state=42, max_depth=5)
    fresh = fresh_model(original)
    
    assert fresh is not original, "Fresh model should be a different instance"
    assert isinstance(fresh, DecisionTreeClassifier), "Fresh model should be DecisionTreeClassifier type"
    assert fresh.random_state == 42, f"random_state should be preserved (got {fresh.random_state})"
    assert fresh.max_depth == 5, f"max_depth should be preserved (got {fresh.max_depth})"
    assert not hasattr(fresh, 'tree_'), "Fresh model should be unfitted (no tree_ attribute)"


def test_fresh_model_sklearn_rf():
    """Test fresh_model with sklearn RandomForest"""
    original = RandomForestClassifier(random_state=42, n_estimators=50)
    fresh = fresh_model(original)
    
    assert fresh is not original, "Fresh model should be a different instance"
    assert isinstance(fresh, RandomForestClassifier), "Fresh model should be RandomForestClassifier type"
    assert fresh.random_state == 42, f"random_state should be preserved (got {fresh.random_state})"
    assert fresh.n_estimators == 50, f"n_estimators should be preserved (got {fresh.n_estimators})"
    assert not hasattr(fresh, 'estimators_'), "Fresh model should be unfitted (no estimators_ attribute)"


def test_fresh_model_preserves_parameters():
    """Test that fresh_model preserves all parameters"""
    original = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced',
        C=0.1
    )
    fresh = fresh_model(original)
    
    assert fresh.random_state == 42, f"random_state should be preserved (got {fresh.random_state})"
    assert fresh.max_iter == 1000, f"max_iter should be preserved (got {fresh.max_iter})"
    assert fresh.class_weight == 'balanced', f"class_weight should be preserved (got {fresh.class_weight})"
    assert fresh.C == 0.1, f"C should be preserved (got {fresh.C})"


def test_fresh_model_unfitted_after_fitting(synthetic_small_data):
    """Test that fresh_model creates unfitted model even from fitted one"""
    X, y = synthetic_small_data
    # Fit original model
    original = LogisticRegression(random_state=42)
    original.fit(X, y)
    
    assert hasattr(original, 'classes_'), "Original model should be fitted after fit()"
    
    # Create fresh model
    fresh = fresh_model(original)
    
    assert fresh is not original, "Fresh model should be a different instance"
    assert not hasattr(fresh, 'classes_'), "Fresh model should be unfitted (no classes_ attribute)"
    assert isinstance(fresh, LogisticRegression), "Fresh model should be LogisticRegression"


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
        
        assert fresh is not original, "Fresh model should be a different instance"
        assert isinstance(fresh, CNNTabularClassifier), "Fresh model should be CNNTabularClassifier type"
        assert fresh.learning_rate == 0.001, f"learning_rate should be preserved (got {fresh.learning_rate})"
        assert fresh.batch_size == 32, f"batch_size should be preserved (got {fresh.batch_size})"
        assert fresh.random_state == 42, f"random_state should be preserved (got {fresh.random_state})"
        assert fresh.hidden_dims == [64, 32], f"hidden_dims should be preserved (got {fresh.hidden_dims})"
        # Model should not be initialized yet
        assert fresh.model is None or isinstance(fresh.model, type(None)), "Model should not be initialized before fit()"
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
        
        assert fresh is not original, "Fresh model should be a different instance"
        assert isinstance(fresh, TabNetClassifierWrapper), "Fresh model should be TabNetClassifierWrapper type"
        assert fresh.n_d == 8, f"n_d should be preserved (got {fresh.n_d})"
        assert fresh.n_a == 8, f"n_a should be preserved (got {fresh.n_a})"
        assert fresh.seed == 42, f"seed should be preserved (got {fresh.seed})"
        assert fresh.max_epochs == 50, f"max_epochs should be preserved (got {fresh.max_epochs})"
        # Model should not be initialized yet
        assert fresh.model is None, "Model should not be initialized before fit()"
    except ImportError:
        pytest.skip("TabNet not available")


def test_fresh_model_independence(synthetic_small_data):
    """Test that fresh model is independent - fitting one doesn't affect the other"""
    X, y = synthetic_small_data
    
    original = LogisticRegression(random_state=42)
    fresh = fresh_model(original)
    
    # Fit fresh model
    fresh.fit(X, y)
    
    # Original should still be unfitted
    assert not hasattr(original, 'classes_'), "Original model should remain unfitted after fitting fresh model"
    
    # Fresh should be fitted
    assert hasattr(fresh, 'classes_'), "Fresh model should be fitted after fit()"
