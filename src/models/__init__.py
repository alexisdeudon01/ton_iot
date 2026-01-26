from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.models.logistic_regression import LogisticRegressionModel
from src.models.decision_tree import DecisionTreeModel
from src.models.random_forest import RandomForestModel
from src.models.cnn import CNNClassifier
from src.models.tabnet import TabNetClassifier
from src.models.evaluator import evaluate_model, build_results_frames


MODEL_REGISTRY = {
    "LR": LogisticRegressionModel,
    "DT": DecisionTreeModel,
    "RF": RandomForestModel,
    "CNN": CNNClassifier,
    "TabNet": TabNetClassifier,
}


def _load_feature_order(processed_dir: Path) -> List[str]:
    path = processed_dir / "feature_order.json"
    return json.loads(path.read_text(encoding="utf-8"))


def train_and_evaluate_all(config: Dict) -> Dict[str, Path]:
    processed_dir = Path("data/processed")
    results_dir = Path("results")
    models_dir = results_dir / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    cic_path = processed_dir / "cic_processed.csv"
    ton_path = processed_dir / "ton_processed.csv"

    if not cic_path.exists() or not ton_path.exists():
        raise FileNotFoundError("Preprocessed datasets are missing. Run preprocessing first.")

    cic_df = pd.read_csv(cic_path)
    ton_df = pd.read_csv(ton_path)
    feature_order = _load_feature_order(processed_dir)

    X_train = cic_df[feature_order].to_numpy()
    y_train = cic_df["y"].to_numpy()
    X_test = ton_df[feature_order].to_numpy()
    y_test = ton_df["y"].to_numpy()

    thresholds = config.get("sme_thresholds", config.get("thresholds", {}))

    results = {}
    for algo_key, model_cls in MODEL_REGISTRY.items():
        model = model_cls(feature_order)
        model.fit(X_train, y_train)
        model_path = models_dir / f"{algo_key.lower()}_model"
        model.save(str(model_path))
        results[algo_key] = evaluate_model(model, algo_key, X_test, y_test, thresholds)

    metrics_df, decision_df = build_results_frames(results)
    metrics_path = results_dir / "metrics.csv"
    decision_path = results_dir / "decision_matrix.csv"
    metrics_df.to_csv(metrics_path, index=False)
    decision_df.to_csv(decision_path, index=False)

    return {
        "metrics": metrics_path,
        "decision_matrix": decision_path,
    }


__all__ = [
    "train_and_evaluate_all",
    "LogisticRegressionModel",
    "DecisionTreeModel",
    "RandomForestModel",
    "CNNClassifier",
    "TabNetClassifier",
]
