import numpy as np
import pandas as pd
from typing import List

class TOPSIS:
    """
    Technique for Order of Preference by Similarity to Ideal Solution.
    Classe les solutions en fonction de leur proximité à la solution idéale positive 
    et de leur éloignement de la solution idéale négative.
    """
    @staticmethod
    def rank(data: np.ndarray, weights: np.ndarray, impacts: List[bool]) -> np.ndarray:
        """
        impacts: True pour maximiser, False pour minimiser.
        """
        # 1. Normalisation (Vector Normalization)
        # On ajoute un petit epsilon pour éviter la division par zéro
        norm_data = data / (np.sqrt((data**2).sum(axis=0)) + 1e-9)
        
        # 2. Pondération
        weighted_data = norm_data * weights
        
        # 3. Solutions idéales (PIS et NIS)
        best_ideal = []
        worst_ideal = []
        for i in range(len(impacts)):
            if impacts[i]: # Maximiser
                best_ideal.append(weighted_data[:, i].max())
                worst_ideal.append(weighted_data[:, i].min())
            else: # Minimiser
                best_ideal.append(weighted_data[:, i].min())
                worst_ideal.append(weighted_data[:, i].max())
        
        best_ideal = np.array(best_ideal)
        worst_ideal = np.array(worst_ideal)
        
        # 4. Distances euclidiennes
        dist_best = np.sqrt(((weighted_data - best_ideal)**2).sum(axis=1))
        dist_worst = np.sqrt(((weighted_data - worst_ideal)**2).sum(axis=1))
        
        # 5. Score de proximité relative
        score = dist_worst / (dist_best + dist_worst + 1e-9)
        return score
