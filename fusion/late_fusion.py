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

    @staticmethod
    def average_cic_ton(proba_cic: np.ndarray, proba_ton: np.ndarray) -> np.ndarray:
        """
        Late Fusion simple : moyenne des probas CIC et TON.
        """
        if proba_cic.shape != proba_ton.shape:
            raise ValueError("Proba CIC et TON doivent avoir la meme shape.")
        return (proba_cic + proba_ton) / 2.0

    @staticmethod
    def late_fusion_predict(proba_cic: np.ndarray, proba_ton: np.ndarray, threshold: float = 0.5):
        """
        Retourne les probas fusionnees + prediction binaire.
        """
        proba_fused = LateFusion.average_cic_ton(proba_cic, proba_ton)
        y_pred = LateFusion.get_decision(proba_fused, threshold=threshold)
        return y_pred, proba_fused
