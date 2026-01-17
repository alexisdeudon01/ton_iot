"""
Tests for explainability.py - SHAP, LIME, native interpretability scores
"""
import sys
from pathlib import Path
import pytest
import numpy as np

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.evaluation.explainability import (
    get_native_interpretability_score,
    compute_shap_score,
    compute_lime_score,
    compute_explainability_score
)

pytest.importorskip("shap", reason="shap not available")
pytest.importorskip("lime", reason="lime not available")


def test_get_native_interpretability_score():
    """Test get_native_interpretability_score for different models"""
    # Highly interpretable models
    assert get_native_interpretability_score('Logistic_Regression') == 1.0
    assert get_native_interpretability_score('Decision_Tree') == 1.0
    assert get_native_interpretability_score('Random_Forest') == 1.0
    
    # Black box models
    assert get_native_interpretability_score('CNN') == 0.0
    assert get_native_interpretability_score('TabNet') == 0.0
    
    # Unknown model
    assert get_native_interpretability_score('UnknownModel') == 0.5


def test_compute_shap_score_tree_model():
    """Test compute_shap_score with tree-based model"""
    from sklearn.ensemble import RandomForestClassifier
    
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.random.randint(0, 2, 200)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    X_sample = X[:50]
    shap_score = compute_shap_score(model, X_sample, top_k=5)
    
    assert shap_score is not None
    assert isinstance(shap_score, float)
    assert shap_score >= 0


def test_compute_shap_score_linear_model():
    """Test compute_shap_score with linear model (KernelExplainer)"""
    from sklearn.linear_model import LogisticRegression
    
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.random.randint(0, 2, 200)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    
    X_sample = X[:50]
    shap_score = compute_shap_score(model, X_sample, top_k=5)
    
    assert shap_score is not None
    assert isinstance(shap_score, float)
    assert shap_score >= 0


def test_compute_shap_score_failure_handling():
    """Test compute_shap_score handles failures gracefully"""
    # Create a model that might fail SHAP computation
    class BadModel:
        def predict_proba(self, X):
            raise ValueError("Error")
    
    model = BadModel()
    X_sample = np.random.randn(10, 5)
    
    shap_score = compute_shap_score(model, X_sample)
    
    assert shap_score is None


def test_compute_lime_score():
    """Test compute_lime_score"""
    from sklearn.ensemble import RandomForestClassifier
    
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.random.randint(0, 2, 200)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    X_sample = X[:50]
    y_sample = y[:50]
    
    lime_score = compute_lime_score(model, X_sample, y_sample, top_k=5)
    
    assert lime_score is not None
    assert isinstance(lime_score, float)
    assert lime_score >= 0


def test_compute_lime_score_failure_handling():
    """Test compute_lime_score handles failures gracefully"""
    class BadModel:
        def predict_proba(self, X):
            raise ValueError("Error")
    
    model = BadModel()
    X_sample = np.random.randn(10, 5)
    y_sample = np.random.randint(0, 2, 10)
    
    lime_score = compute_lime_score(model, X_sample, y_sample)
    
    assert lime_score is None


def test_compute_explainability_score_all_components():
    """Test compute_explainability_score with all components"""
    result = compute_explainability_score(
        model_name='Random_Forest',
        shap_score=0.5,
        lime_score=0.3,
        weights=(0.5, 0.3, 0.2)
    )
    
    assert 'explain_score' in result
    assert 'native_score' in result
    assert 'shap_score' in result
    assert 'lime_score' in result
    assert 'weights_used' in result
    assert 'missing_components' in result
    
    assert result['native_score'] == 1.0
    assert result['shap_score'] == 0.5
    assert result['lime_score'] == 0.3
    assert result['missing_components'] == []
    assert 0 <= result['explain_score'] <= 1


def test_compute_explainability_score_missing_shap():
    """Test compute_explainability_score with missing SHAP score"""
    result = compute_explainability_score(
        model_name='Logistic_Regression',
        shap_score=None,
        lime_score=0.4,
        weights=(0.5, 0.3, 0.2)
    )
    
    assert 'shap' in result['missing_components']
    assert np.isnan(result['shap_score'])
    assert result['lime_score'] == 0.4
    # Weights should be renormalized (native + lime only)
    assert sum(result['weights_used'].values()) == pytest.approx(1.0)


def test_compute_explainability_score_missing_lime():
    """Test compute_explainability_score with missing LIME score"""
    result = compute_explainability_score(
        model_name='CNN',
        shap_score=0.5,
        lime_score=None,
        weights=(0.5, 0.3, 0.2)
    )
    
    assert 'lime' in result['missing_components']
    assert np.isnan(result['lime_score'])
    assert result['shap_score'] == 0.5
    assert result['native_score'] == 0.0
    assert sum(result['weights_used'].values()) == pytest.approx(1.0)


def test_compute_explainability_score_native_only():
    """Test compute_explainability_score with only native score"""
    result = compute_explainability_score(
        model_name='Decision_Tree',
        shap_score=None,
        lime_score=None,
        weights=(0.5, 0.3, 0.2)
    )
    
    assert len(result['missing_components']) == 2
    assert 'shap' in result['missing_components']
    assert 'lime' in result['missing_components']
    assert result['native_score'] == 1.0
    # With only native, weight should be 1.0
    assert result['weights_used']['native'] == pytest.approx(1.0)


def test_compute_explainability_score_nan_handling():
    """Test compute_explainability_score handles NaN values"""
    result = compute_explainability_score(
        model_name='Random_Forest',
        shap_score=np.nan,
        lime_score=0.3,
        weights=(0.5, 0.3, 0.2)
    )
    
    assert 'shap' in result['missing_components']
    assert np.isnan(result['shap_score'])
    assert result['lime_score'] == 0.3
