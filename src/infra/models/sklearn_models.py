import joblib
import numpy as np
from typing import List, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.core.ports.models import ModelPort

class SklearnModel(ModelPort):
    def __init__(self, model_type: str, feature_order: List[str], **kwargs):
        self._model_type = model_type
        self._feature_order = feature_order
        if model_type == "LR":
            self.model = LogisticRegression(**kwargs)
        elif model_type == "DT":
            self.model = DecisionTreeClassifier(**kwargs)
        elif model_type == "RF":
            self.model = RandomForestClassifier(**kwargs)
        else:
            raise ValueError(f"Unsupported sklearn model type: {model_type}")

    def train(self, X: Any, y: Any, **kwargs) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "feature_order": self._feature_order, "type": self._model_type}, path)

    def load(self, path: str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self._feature_order = data["feature_order"]
        self._model_type = data["type"]

    @property
    def feature_order(self) -> List[str]:
        return self._feature_order
