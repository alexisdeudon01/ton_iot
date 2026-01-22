import numpy as np
from typing import List

class LateFusion:
    """
    Fusion des probabilités de plusieurs modèles (Late Fusion).
    Formule : p_fusion = w * p_CIC + (1-w) * p_TON
    """
    @staticmethod
    def weighted_average(probas_list: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """
        Calcule la moyenne pondérée des probabilités de classe.
        """
        if len(probas_list) != len(weights):
            raise ValueError("Mismatch entre nombre de modèles et nombre de poids.")
            
        # Normalisation des poids
        w_norm = np.array(weights) / np.sum(weights)
        
        # Initialisation avec la forme du premier tableau de probas
        fused_proba = np.zeros_like(probas_list[0])
        
        for p, w in zip(probas_list, w_norm):
            fused_proba += p * w
            
        return fused_proba

    @staticmethod
    def get_decision(fused_proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        ŷ = 1 si p_fusion[DDoS] ≥ threshold
        """
        # On suppose que la colonne 1 est la classe DDoS (y=1)
        return (fused_proba[:, 1] >= threshold).astype(int)
