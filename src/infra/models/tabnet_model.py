import numpy as np
from typing import List, Any, Optional
from src.core.ports.models import ModelPort

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

class TabNetModel(ModelPort):
    def __init__(self, feature_order: List[str], **kwargs):
        self._feature_order = feature_order
        self.model = None
        self.kwargs = kwargs
        if TABNET_AVAILABLE:
            self.model = TabNetClassifier(**kwargs)
        else:
            # VERIFY: pytorch-tabnet is not installed. 
            # Alternatives: 1) pip install pytorch-tabnet 2) Use RF/CNN instead.
            pass

    def train(self, X: Any, y: Any, **kwargs) -> None:
        if not TABNET_AVAILABLE:
            raise RuntimeError("TabNetClassifier is not available. Install pytorch-tabnet.")
        
        # TabNet expects numpy arrays
        X_np = X.astype(np.float32) if hasattr(X, "astype") else np.array(X, dtype=np.float32)
        y_np = y.astype(np.int64) if hasattr(y, "astype") else np.array(y, dtype=np.int64)
        
        self.model.fit(
            X_train=X_np, y_train=y_np,
            max_epochs=kwargs.get("max_epochs", 20),
            patience=kwargs.get("patience", 5),
            batch_size=kwargs.get("batch_size", 1024),
            virtual_batch_size=kwargs.get("virtual_batch_size", 128),
            num_workers=0,
            drop_last=False
        )

    def predict_proba(self, X: Any) -> np.ndarray:
        if not TABNET_AVAILABLE:
            raise RuntimeError("TabNetClassifier is not available.")
        X_np = X.astype(np.float32) if hasattr(X, "astype") else np.array(X, dtype=np.float32)
        return self.model.predict_proba(X_np)

    def save(self, path: str) -> None:
        if not TABNET_AVAILABLE:
            return
        self.model.save_model(path)

    def load(self, path: str) -> None:
        if not TABNET_AVAILABLE:
            raise RuntimeError("TabNetClassifier is not available.")
        self.model = TabNetClassifier()
        self.model.load_model(path)

    @property
    def feature_order(self) -> List[str]:
        return self._feature_order
