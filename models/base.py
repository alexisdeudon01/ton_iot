from abc import ABC, abstractmethod
import numpy as np
from typing import Any, List

class AbstractModel(ABC):
    """
    Interface de base pour tous les modèles de détection DDoS.
    Garantit une signature cohérente pour l'entraînement et l'inférence.
    """
    def __init__(self, name: str, feature_order: List[str]):
        self.name = name
        self.feature_order = feature_order
        self.model = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Entraîne le modèle sur les données fournies."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retourne les probabilités pour chaque classe."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Sauvegarde l'état du modèle."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Charge l'état du modèle."""
        pass
