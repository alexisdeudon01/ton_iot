from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from models.lr import LRModel
from models.dt import DTModel
from models.rf import RFModel
from models.cnn import CNNModel
from models.tabnet import TabNetModel


MODEL_REGISTRY = {
    "LR": LRModel,
    "DT": DTModel,
    "RF": RFModel,
    "CNN": CNNModel,
    "TabNet": TabNetModel,
}


def _model_filename(algo: str) -> str:
    if algo == "CNN":
        return "model_cnn.pt"
    if algo == "TabNet":
        return "model_tabnet"
    return f"model_{algo.lower()}.joblib"


def train_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_order: List[str],
    algorithms: Iterable[str],
    output_dir: str | Path,
) -> Tuple[Dict[str, object], Dict[str, Path]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trained: Dict[str, object] = {}
    model_paths: Dict[str, Path] = {}

    for algo in algorithms:
        if algo not in MODEL_REGISTRY:
            raise ValueError(f"Unknown algorithm key: {algo}")
        model_cls = MODEL_REGISTRY[algo]
        model = model_cls(feature_order)
        model.fit(X, y)
        model_path = output_dir / _model_filename(algo)
        model.save(str(model_path))
        trained[algo] = model
        model_paths[algo] = model_path

    return trained, model_paths
