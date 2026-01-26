import os
from typing import List

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

from models.base import AbstractModel

class TabNetModel(AbstractModel):
    """
    Adaptateur pour le modèle TabNet (Deep Learning pour données tabulaires).
    """
    def __init__(self, feature_order: List[str]):
        super().__init__("TabNet", feature_order)
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TabNetClassifier(
            n_d=8, n_a=8, n_steps=3,
            gamma=1.3, n_independent=2, n_shared=2,
            lambda_sparse=1e-3, momentum=0.02, clip_value=2.,
            optimizer_fn=None, # Default Adam
            optimizer_params=dict(lr=2e-2),
            scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.9),
            scheduler_fn=None,
            mask_type='sparsemax',
            verbose=0,
            seed=42,
            device_name=device_name
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        # En mode test ou petit volume, on réduit les epochs
        self.model.fit(
            X_train=X, y_train=y,
            max_epochs=20,
            patience=5,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str):
        # TabNet save_model crée un fichier .zip
        self.model.save_model(path)

    def load(self, path: str):
        # On s'assure de charger le .zip si nécessaire
        if not path.endswith(".zip") and not os.path.exists(path):
            path = path + ".zip"
        self.model.load_model(path)
