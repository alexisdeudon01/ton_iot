import polars as pl
import hdbscan
import numpy as np
from typing import Tuple, Dict

class HDBSCANModule:
    """
    Module de clustering non supervisé pour identifier des patterns d'attaque.
    """
    def __init__(self, min_cluster_size: int = 50, min_samples: int = 10):
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=True
        )

    def fit_predict(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Exécute le clustering et retourne les labels et les statistiques.
        """
        labels = self.clusterer.fit_predict(data)
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        stats = {
            "n_clusters": len(unique_labels[unique_labels != -1]),
            "noise_points": int(counts[unique_labels == -1][0]) if -1 in unique_labels else 0,
            "cluster_distribution": {str(l): int(c) for l, c in zip(unique_labels, counts)}
        }
        
        return labels, stats
