import numpy as np
from scipy.stats import entropy

class ComplexityEvaluator:
    """
    Mesure la complexité d'une explication via l'entropie de Shannon.
    H = − Σ pᵢ log(pᵢ)
    """
    @staticmethod
    def compute(shap_values: np.ndarray) -> float:
        """
        Calcule l'entropie des valeurs SHAP absolues.
        """
        abs_vals = np.abs(shap_values)
        total = np.sum(abs_vals)
        if total == 0: return 0.0
        
        probs = abs_vals / total
        # Filtrer les zéros pour le log
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))
