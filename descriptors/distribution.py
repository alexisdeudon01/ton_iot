import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from typing import List

class DistributionComparator:
    """
    Implémente les mesures de similarité entre distributions pour l'alignement des caractéristiques.
    """
    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calcule la similarité cosinus entre deux vecteurs de descripteurs.
        cos(x,y) = (x·y)/(||x||·||y||)
        """
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    @staticmethod
    def ks_test(s1: np.ndarray, s2: np.ndarray) -> float:
        """
        Calcule la p-value du test de Kolmogorov-Smirnov pour deux échantillons.
        D = sup |F₁(x) − F₂(x)|
        """
        if len(s1) == 0 or len(s2) == 0:
            return 0.0
        # ks_2samp renvoie (statistic, pvalue)
        return float(ks_2samp(s1, s2).pvalue)

    @staticmethod
    def wasserstein(s1: np.ndarray, s2: np.ndarray) -> float:
        """
        Calcule la distance de Wasserstein (Earth Mover's Distance).
        W(P,Q) = ∫ |F₁(x) − F₂(x)| dx
        """
        if len(s1) == 0 or len(s2) == 0:
            return 1e9
        return float(wasserstein_distance(s1, s2))
