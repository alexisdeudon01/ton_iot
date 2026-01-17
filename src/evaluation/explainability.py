#!/usr/bin/env python3
"""
Dimension 3: Explainability Metrics (native, SHAP, LIME)
"""
import numpy as np
import logging
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)

import shap
import lime
import lime.lime_tabular

SHAP_AVAILABLE = True
LIME_AVAILABLE = True


def get_native_interpretability_score(model_name: str) -> float:
    """
    Get native interpretability score (0-1)

    Scoring:
    - Decision Tree / Random Forest: 1.0 (highly interpretable)
    - Logistic Regression: 1.0 (linear, interpretable)
    - CNN / TabNet: 0.0 (black box)

    Args:
        model_name: Model name

    Returns:
        Native interpretability score (0-1)
    """
    native_scores = {
        'Logistic_Regression': 1.0,
        'Decision_Tree': 1.0,
        'Random_Forest': 1.0,
        'CNN': 0.0,
        'TabNet': 0.0
    }
    return native_scores.get(model_name, 0.5)


def compute_shap_score(model, X_sample, top_k: int = 10) -> Optional[float]:
    """
    Compute SHAP importance score (mean absolute SHAP values)

    Args:
        model: Trained model
        X_sample: Sample data for SHAP computation
        top_k: Number of top features to consider

    Returns:
        SHAP score (mean absolute importance)
    """
    try:
        explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.KernelExplainer(model.predict_proba, X_sample[:100])
        shap_values = explainer.shap_values(X_sample[:min(100, len(X_sample))])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Binary classification: use positive class
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        top_k_mean = np.mean(np.sort(mean_abs)[-top_k:])
        return float(top_k_mean)
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return None


def compute_lime_score(model, X_sample, y_sample, top_k: int = 10) -> Optional[float]:
    """
    Compute LIME importance score (mean absolute LIME importance)

    Args:
        model: Trained model
        X_sample: Sample data
        y_sample: Sample labels
        top_k: Number of top features

    Returns:
        LIME score (mean absolute importance)
    """
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(X_sample, mode='classification')
        importances = []
        for i in range(min(10, len(X_sample))):
            exp = explainer.explain_instance(X_sample[i], model.predict_proba)
            top_features = exp.as_list()[:top_k]
            abs_imp = [abs(v) for _, v in top_features]
            importances.extend(abs_imp)
        return float(np.mean(importances)) if importances else None
    except Exception as e:
        logger.warning(f"LIME computation failed: {e}")
        return None


def compute_explainability_score(model_name: str, shap_score: Optional[float] = None,
                                  lime_score: Optional[float] = None,
                                  weights=(0.5, 0.3, 0.2)) -> Dict[str, Any]:
    """
    Compute explainability score with normalization if SHAP/LIME missing

    Args:
        model_name: Model name
        shap_score: SHAP score (can be None)
        lime_score: LIME score (can be None)
        weights: (w_native, w_shap, w_lime)

    Returns:
        Dict with explain_score, native_score, shap_score, lime_score,
        weights_used, missing_components
    """
    w_native, w_shap, w_lime = weights
    native_score = get_native_interpretability_score(model_name)

    # Normalize weights if SHAP/LIME missing
    available_weights = {'native': w_native}
    missing = []

    if shap_score is None or np.isnan(shap_score):
        missing.append('shap')
        shap_score = None
    else:
        available_weights['shap'] = w_shap

    if lime_score is None or np.isnan(lime_score):
        missing.append('lime')
        lime_score = None
    else:
        available_weights['lime'] = w_lime

    # Renormalize weights
    total_weight = sum(available_weights.values())
    if total_weight > 0:
        for k in available_weights:
            available_weights[k] /= total_weight

    # Normalize SHAP/LIME to [0,1] if available (assume already in reasonable range, or min-max)
    # For simplicity, assume they're already normalized or need min-max scaling
    # In practice, you'd normalize across all models

    # Compute weighted score
    score = available_weights['native'] * native_score
    if shap_score is not None:
        score += available_weights.get('shap', 0) * min(shap_score, 1.0)
    if lime_score is not None:
        score += available_weights.get('lime', 0) * min(lime_score, 1.0)

    return {
        'explain_score': float(score),
        'native_score': float(native_score),
        'shap_score': float(shap_score) if shap_score is not None else np.nan,
        'lime_score': float(lime_score) if lime_score is not None else np.nan,
        'weights_used': available_weights,
        'missing_components': missing
    }
