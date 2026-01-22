import shap
import numpy as np
from typing import Any, Callable

class ShapExplainer:
    """
    Génère des explications SHAP pour les modèles de détection DDoS.
    """
    def __init__(self, model_predict_fn: Callable, background_data: np.ndarray):
        # Utilisation de KernelExplainer pour une compatibilité maximale (LR, CNN, etc.)
        self.explainer = shap.KernelExplainer(model_predict_fn, background_data)

    def explain(self, X: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """
        Calcule les SHAP values pour X.
        """
        shap_values = self.explainer.shap_values(X, n_samples=n_samples)
        # shap_values est souvent une liste [probas_0, probas_1] pour la classification binaire
        if isinstance(shap_values, list):
            return shap_values[1] # On explique la classe DDoS (y=1)
        return shap_values

    @staticmethod
    def compute_global_importance(shap_values: np.ndarray) -> np.ndarray:
        """
        GlobalSHAPᵢ = (1/N) Σ |φᵢ|
        """
        return np.mean(np.abs(shap_values), axis=0)
