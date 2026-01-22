from sklearn.ensemble import RandomForestClassifier
from models.base import AbstractModel
import joblib
import numpy as np
from typing import List

class RFModel(AbstractModel):
    """
    Implémentation du modèle Random Forest pour la détection DDoS.
    """
    def __init__(self, feature_order: List[str]):
        super().__init__("RandomForest", feature_order)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
