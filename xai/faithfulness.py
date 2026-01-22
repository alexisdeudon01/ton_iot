import numpy as np
from typing import Callable

class FaithfulnessEvaluator:
    """
    Évalue la fidélité des explications (Faithfulness).
    Δ = Perf_ref − Perf_ablated
    """
    @staticmethod
    def compute(X: np.ndarray, 
                shap_values: np.ndarray, 
                predict_fn: Callable, 
                metric_fn: Callable) -> float:
        """
        Calcule la fidélité en supprimant les features les plus importantes.
        """
        # Performance de référence
        y_proba_ref = predict_fn(X)
        perf_ref = metric_fn(y_proba_ref)
        
        # Ablation des top features (mise à zéro)
        X_ablated = X.copy()
        # On identifie les indices des 3 features les plus importantes globalement
        top_features = np.argsort(np.abs(shap_values))[-3:]
        X_ablated[:, top_features] = 0
        
        y_proba_ablated = predict_fn(X_ablated)
        perf_ablated = metric_fn(y_proba_ablated)
        
        return float(perf_ref - perf_ablated)
