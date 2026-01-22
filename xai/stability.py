import numpy as np
from typing import Callable
from xai.shap_values import ShapExplainer
from descriptors.distribution import DistributionComparator

class StabilityEvaluator:
    """
    Mesure la stabilité des explications SHAP sous perturbation.
    Stability = cos(φ(x), φ(x+ε))
    """
    def __init__(self, explainer: ShapExplainer):
        self.explainer = explainer

    def compute(self, x: np.ndarray, epsilon: float = 0.01) -> float:
        """
        Calcule la stabilité pour une instance x.
        """
        phi_x = self.explainer.explain(x.reshape(1, -1))
        
        # Perturbation gaussienne légère
        x_eps = x + np.random.normal(0, epsilon, x.shape)
        phi_x_eps = self.explainer.explain(x_eps.reshape(1, -1))
        
        return DistributionComparator.cosine_similarity(phi_x.flatten(), phi_x_eps.flatten())
