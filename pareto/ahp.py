import numpy as np
from typing import List

class AHP:
    """
    Analytic Hierarchy Process pour la pondération experte des objectifs.
    Permet de transformer des comparaisons par paires en poids numériques.
    """
    @staticmethod
    def compute_weights(comparison_matrix: np.ndarray) -> np.ndarray:
        """
        Calcule le vecteur propre principal pour obtenir les poids.
        """
        eigvals, eigvecs = np.linalg.eig(comparison_matrix)
        max_eigval_idx = np.argmax(eigvals.real)
        weights = eigvecs[:, max_eigval_idx].real
        return weights / np.sum(weights)

    @staticmethod
    def consistency_ratio(comparison_matrix: np.ndarray, weights: np.ndarray) -> float:
        """
        Calcule le ratio de cohérence de la matrice experte.
        Un ratio < 0.1 est généralement considéré comme acceptable.
        """
        n = len(weights)
        if n <= 2: return 0.0
        
        lambda_max = np.real(np.max(np.linalg.eigvals(comparison_matrix)))
        ci = (lambda_max - n) / (n - 1)
        # Random Index values for n=1..8
        ri_values = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}
        ri = ri_values.get(n, 1.45)
        return ci / ri
