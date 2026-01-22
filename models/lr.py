from sklearn.linear_model import LogisticRegression
from models.base import AbstractModel
import joblib
import numpy as np
from typing import List

class LRModel(AbstractModel):
    def __init__(self, feature_order: List[str]):
        super().__init__("LogisticRegression", feature_order)
        self.model = LogisticRegression(max_iter=1000, random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
