from sklearn.tree import DecisionTreeClassifier
from models.base import AbstractModel
import joblib
import numpy as np
from typing import List

class DTModel(AbstractModel):
    """
    Implémentation du modèle Decision Tree pour la détection DDoS.
    """
    def __init__(self, feature_order: List[str]):
        super().__init__("DecisionTree", feature_order)
        self.model = DecisionTreeClassifier(random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
