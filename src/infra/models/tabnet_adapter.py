import polars as pl
from pytorch_tabnet.tab_model import TabNetClassifier
import os
from typing import Dict, Any
from src.core.ports.interfaces import IModelAdapter

class TabNetAdapter(IModelAdapter):
    def train(self, X: pl.DataFrame, y: pl.Series, params: Dict[str, Any]) -> Any:
        model = TabNetClassifier(**params.get("model_params", {}))
        
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        
        model.fit(
            X_train=X_np, y_train=y_np,
            max_epochs=params.get("epochs", 10),
            patience=params.get("patience", 5),
            batch_size=params.get("batch_size", 1024),
            virtual_batch_size=params.get("virtual_batch_size", 128),
            num_workers=0,
            drop_last=False
        )
        return model

    def predict_proba(self, model: Any, X: pl.DataFrame) -> pl.Series:
        probas = model.predict_proba(X.to_numpy())[:, 1]
        return pl.Series("proba", probas)

    def save(self, model: Any, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # TabNet save creates a zip file
        model.save_model(path)

    def load(self, path: str) -> Any:
        model = TabNetClassifier()
        model.load_model(path)
        return model
