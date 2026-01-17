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
    score_lr = get_native_interpretability_score('Logistic_Regression')
    assert score_lr == 1.0, f"Logistic_Regression should score 1.0 (got {score_lr})"
    
    score_dt = get_native_interpretability_score('Decision_Tree')
    assert score_dt == 1.0, f"Decision_Tree should score 1.0 (got {score_dt})"
    
    score_rf = get_native_interpretability_score('Random_Forest')
    assert score_rf == 1.0, f"Random_Forest should score 1.0 (got {score_rf})"
    
    # Black box models
    score_cnn = get_native_interpretability_score('CNN')
    assert score_cnn == 0.0, f"CNN should score 0.0 (got {score_cnn})"
    
    score_tabnet = get_native_interpretability_score('TabNet')
    assert score_tabnet == 0.0, f"TabNet should score 0.0 (got {score_tabnet})"
    
    # Unknown model
    score_unknown = get_native_interpretability_score('UnknownModel')
    assert score_unknown == 0.5, f"UnknownModel should score 0.5 (got {score_unknown})"


def test_compute_shap_score_tree_model(synthetic_binary_data):
    """Test compute_shap_score with tree-based model"""
    from sklearn.ensemble import RandomForestClassifier
    
    X, y = synthetic_binary_data
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    X_sample = X[:50]
    shap_score = compute_shap_score(model, X_sample, top_k=5)
    
    assert shap_score is not None, "SHAP score should not be None"
    assert isinstance(shap_score, float), f"SHAP score should be float (got {type(shap_score)})"
    assert shap_score >= 0, f"SHAP score should be non-negative (got {shap_score})"


def test_compute_shap_score_linear_model(synthetic_binary_data):
    """Test compute_shap_score with linear model (KernelExplainer)"""
    from sklearn.linear_model import LogisticRegression
    
    X, y = synthetic_binary_data
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    
    X_sample = X[:50]
    shap_score = compute_shap_score(model, X_sample, top_k=5)
    
    assert shap_score is not None, "SHAP score should not be None"
    assert isinstance(shap_score, float), f"SHAP score should be float (got {type(shap_score)})"
    assert shap_score >= 0, f"SHAP score should be non-negative (got {shap_score})"


def test_compute_shap_score_failure_handling():
    """Test compute_shap_score handles failures gracefully"""
    # Create a model that might fail SHAP computation
    class BadModel:
        def predict_proba(self, X):
            raise ValueError("Error")
    
    model = BadModel()
    X_sample = np.random.randn(10, 5)
    
    shap_score = compute_shap_score(model, X_sample)
    
    assert shap_score is None, "SHAP score should be None when computation fails (error handling)"


def test_compute_lime_score(synthetic_binary_data):
    """Test compute_lime_score"""
    from sklearn.ensemble import RandomForestClassifier
    
    X, y = synthetic_binary_data
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    X_sample = X[:50]
    y_sample = y[:50]
    
    lime_score = compute_lime_score(model, X_sample, y_sample, top_k=5)
    
    assert lime_score is not None, "LIME score should not be None"
    assert isinstance(lime_score, float), f"LIME score should be float (got {type(lime_score)})"
    assert lime_score >= 0, f"LIME score should be non-negative (got {lime_score})"


def test_compute_lime_score_failure_handling():
    """Test compute_lime_score handles failures gracefully"""
    class BadModel:
        def predict_proba(self, X):
            raise ValueError("Error")
    
    model = BadModel()
    X_sample = np.random.randn(10, 5)
    y_sample = np.random.randint(0, 2, 10)
    
    lime_score = compute_lime_score(model, X_sample, y_sample)
    
    assert lime_score is None, "LIME score should be None when computation fails (error handling)"


def test_compute_explainability_score_all_components():
    """Test compute_explainability_score with all components"""
    result = compute_explainability_score(
        model_name='Random_Forest',
        shap_score=0.5,
        lime_score=0.3,
        weights=(0.5, 0.3, 0.2)
    )
    
    assert 'explain_score' in result, "Result should contain 'explain_score' key"
    assert 'native_score' in result, "Result should contain 'native_score' key"
    assert 'shap_score' in result, "Result should contain 'shap_score' key"
    assert 'lime_score' in result, "Result should contain 'lime_score' key"
    assert 'weights_used' in result, "Result should contain 'weights_used' key"
    assert 'missing_components' in result, "Result should contain 'missing_components' key"
    
    assert result['native_score'] == 1.0, f"native_score should be 1.0 for Random_Forest (got {result['native_score']})"
    assert result['shap_score'] == 0.5, f"shap_score should be 0.5 (got {result['shap_score']})"
    assert result['lime_score'] == 0.3, f"lime_score should be 0.3 (got {result['lime_score']})"
    assert result['missing_components'] == [], "No components should be missing (got {result['missing_components']})"
    assert 0 <= result['explain_score'] <= 1, f"explain_score should be in [0, 1] (got {result['explain_score']})"


def test_compute_explainability_score_missing_shap():
    """Test compute_explainability_score with missing SHAP score"""
    result = compute_explainability_score(
        model_name='Logistic_Regression',
        shap_score=None,
        lime_score=0.4,
        weights=(0.5, 0.3, 0.2)
    )
    
    assert 'shap' in result['missing_components'], "SHAP should be in missing_components"
    assert np.isnan(result['shap_score']), "shap_score should be NaN when missing"
    assert result['lime_score'] == 0.4, f"lime_score should be 0.4 (got {result['lime_score']})"
    # Weights should be renormalized (native + lime only)
    assert sum(result['weights_used'].values()) == pytest.approx(1.0), "Weights should sum to 1.0 after renormalization"


def test_compute_explainability_score_missing_lime():
    """Test compute_explainability_score with missing LIME score"""
    result = compute_explainability_score(
        model_name='CNN',
        shap_score=0.5,
        lime_score=None,
        weights=(0.5, 0.3, 0.2)
    )
    
    assert 'lime' in result['missing_components'], "LIME should be in missing_components"
    assert np.isnan(result['lime_score']), "lime_score should be NaN when missing"
    assert result['shap_score'] == 0.5, f"shap_score should be 0.5 (got {result['shap_score']})"
    assert result['native_score'] == 0.0, f"native_score should be 0.0 for CNN (got {result['native_score']})"
    assert sum(result['weights_used'].values()) == pytest.approx(1.0), "Weights should sum to 1.0 after renormalization"


def test_compute_explainability_score_native_only():
    """Test compute_explainability_score with only native score"""
    result = compute_explainability_score(
        model_name='Decision_Tree',
        shap_score=None,
        lime_score=None,
        weights=(0.5, 0.3, 0.2)
    )
    
    assert len(result['missing_components']) == 2, f"Should have 2 missing components (got {len(result['missing_components'])})"
    assert 'shap' in result['missing_components'], "SHAP should be in missing_components"
    assert 'lime' in result['missing_components'], "LIME should be in missing_components"
    assert result['native_score'] == 1.0, f"native_score should be 1.0 for Decision_Tree (got {result['native_score']})"
    # With only native, weight should be 1.0
    assert result['weights_used']['native'] == pytest.approx(1.0), "Native weight should be 1.0 when only native available"


def test_compute_explainability_score_nan_handling():
    """Test compute_explainability_score handles NaN values"""
    result = compute_explainability_score(
        model_name='Random_Forest',
        shap_score=np.nan,
        lime_score=0.3,
        weights=(0.5, 0.3, 0.2)
    )
    
    assert 'shap' in result['missing_components'], "SHAP should be in missing_components when NaN"
    assert np.isnan(result['shap_score']), "shap_score should be NaN when input is NaN"
    assert result['lime_score'] == 0.3, f"lime_score should be 0.3 (got {result['lime_score']})"
