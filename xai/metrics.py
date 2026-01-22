import numpy as np
from scipy.stats import entropy
from descriptors.distribution import DistributionComparator

class XAIMetrics:
    """
    Objectivation de l'explicabilité via des métriques mathématiques.
    """
    @staticmethod
    def top_k_concentration(shap_importances: np.ndarray, k: int = 5) -> float:
        """
        Top-K SHAP concentration : TopK = Σₖ |φᵢ| / Σ |φᵢ|
        """
        abs_imp = np.abs(shap_importances)
        total = np.sum(abs_imp)
        if total == 0: return 0.0
        
        top_k_val = np.sort(abs_imp)[-k:].sum()
        return float(top_k_val / total)

    @staticmethod
    def stability(shap_v1: np.ndarray, shap_v2: np.ndarray) -> float:
        """
        Stability = cos(φ(x), φ(x+ε))
        Mesure la robustesse de l'explication face à de petites perturbations.
        """
        return DistributionComparator.cosine_similarity(shap_v1, shap_v2)

    @staticmethod
    def complexity(shap_importances: np.ndarray) -> float:
        """
        Entropy SHAP : H = − Σ pᵢ log(pᵢ)
        Mesure si l'explication est diffuse ou concentrée.
        """
        abs_imp = np.abs(shap_importances)
        total = np.sum(abs_imp)
        if total == 0: return 0.0
        
        p = abs_imp / total
        # Filtrer les zéros pour le log
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))
