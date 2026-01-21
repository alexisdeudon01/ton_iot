import polars as pl
import joblib
import os
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.core.ports.interfaces import IModelAdapter

class SklearnAdapter(IModelAdapter):
    def train(self, X: pl.DataFrame, y: pl.Series, params: Dict[str, Any]) -> Any:
        model_type = params.get("model_type")
        model_params = params.get("model_params", {})
        
        if model_type == "LR":
            model = LogisticRegression(**model_params)
        elif model_type == "DT":
            model = DecisionTreeClassifier(**model_params)
        elif model_type == "RF":
            model = RandomForestClassifier(**model_params)
        else:
            raise ValueError(f"Unsupported sklearn model type: {model_type}")
            
        model.fit(X.to_numpy(), y.to_numpy())
        return model

    def predict_proba(self, model: Any, X: pl.DataFrame) -> pl.Series:
        probas = model.predict_proba(X.to_numpy())[:, 1]
        return pl.Series("proba", probas)

    def save(self, model: Any, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)

    def load(self, path: str) -> Any:
        return joblib.load(path)
