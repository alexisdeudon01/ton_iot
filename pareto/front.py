import numpy as np
import pandas as pd
from typing import List

class ParetoFront:
    """
    Identifie les solutions non dominées (Front de Pareto) pour l'optimisation multi-objectifs.
    """
    @staticmethod
    def is_dominated(candidate: np.ndarray, others: np.ndarray) -> bool:
        """
        Vérifie si une solution est dominée par au moins une autre.
        A ≺ B si ∀i zᵢ(A) ≤ zᵢ(B) ∧ ∃j zⱼ(A) < zⱼ(B) (pour maximisation).
        """
        for other in others:
            # On vérifie si 'other' est meilleur ou égal sur tous les points
            # et strictement meilleur sur au moins un point.
            if np.all(other >= candidate) and np.any(other > candidate):
                return True
        return False

    @staticmethod
    def get_pareto_front(metrics_df: pd.DataFrame, objectives: List[str]) -> pd.DataFrame:
        """
        Retourne les lignes du DataFrame qui ne sont pas dominées sur les objectifs donnés.
        Note: Les objectifs de coût (RAM, CPU) doivent être inversés (négatifs) avant appel.
        """
        data = metrics_df[objectives].to_numpy()
        is_pareto = []
        
        for i in range(len(data)):
            candidate = data[i]
            others = np.delete(data, i, axis=0)
            is_pareto.append(not ParetoFront.is_dominated(candidate, others))
            
        return metrics_df[is_pareto].copy()
