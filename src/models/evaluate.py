from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from evaluation.performance import PerformanceEvaluator
from evaluation.resources import ResourceTracker
from models.lr import LRModel
from models.dt import DTModel
from models.rf import RFModel
from models.cnn import CNNModel
from models.tabnet import TabNetModel
from src.mcdm.metrics_utils import compute_f_expl, compute_f_perf, compute_f_res, explainability_metrics


MODEL_REGISTRY = {
    "LR": LRModel,
    "DT": DTModel,
    "RF": RFModel,
    "CNN": CNNModel,
    "TabNet": TabNetModel,
}


@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    decision: Dict[str, float]
    resources: Dict[str, float]


def _estimate_n_params(model_obj) -> int:
    try:
        if hasattr(model_obj, "parameters"):
            return int(sum(p.numel() for p in model_obj.parameters()))
    except Exception:
        pass

    try:
        if hasattr(model_obj, "coef_"):
            return int(np.prod(model_obj.coef_.shape))
    except Exception:
        pass

    try:
        if hasattr(model_obj, "feature_importances_"):
            return int(len(model_obj.feature_importances_))
    except Exception:
        pass

    return 1


def _predict_labels(probas: np.ndarray, threshold: float) -> np.ndarray:
    if probas.ndim == 1:
        return (probas >= threshold).astype(int)
    return (probas[:, 1] >= threshold).astype(int)


def load_model(algo: str, feature_order: List[str], model_path: str | Path):
    if algo not in MODEL_REGISTRY:
        raise ValueError(f"Unknown algorithm key: {algo}")
    model = MODEL_REGISTRY[algo](feature_order)
    model.load(str(model_path))
    return model


def evaluate_model(
    algo: str,
    model,
    X: np.ndarray,
    y: np.ndarray,
    thresholds: Dict[str, float] | None = None,
    threshold_proba: float = 0.5,
) -> EvaluationResult:
    tracker = ResourceTracker()
    tracker.start()
    probas = model.predict_proba(X)
    usage = tracker.get_usage()

    y_pred = _predict_labels(probas, threshold_proba)
    metrics = PerformanceEvaluator.compute_metrics(y, y_pred)

    try:
        auc = roc_auc_score(y, probas[:, 1])
    except Exception:
        auc = 0.0

    n_params = _estimate_n_params(model.model if hasattr(model, "model") else model)
    s_intrinsic, shap_available, n_params, shap_std = explainability_metrics(algo, n_params)

    gap = 0.0
    f_perf = compute_f_perf(metrics["f1"], metrics["recall"], float(auc), gap)
    f_expl = compute_f_expl(s_intrinsic, shap_available, n_params, shap_std)

    tau_memory = None
    tau_cpu = None
    if thresholds:
        mem_mb = thresholds.get("T_memory_mb", None)
        if mem_mb is not None:
            tau_memory = float(mem_mb) * 1024 * 1024
        tau_cpu = thresholds.get("T_cpu_percent", None)
    mem_bytes = usage["ram_mb"] * 1024 * 1024
    f_res = compute_f_res(mem_bytes, usage["cpu_percent"], tau_memory or 524_288_000, tau_cpu or 80.0)

    decision = {
        "f_perf": float(f_perf),
        "f_expl": float(f_expl),
        "f_res": float(f_res),
    }

    metrics["roc_auc"] = float(auc)

    return EvaluationResult(metrics=metrics, decision=decision, resources=usage)


def evaluate_models(
    models: Dict[str, object],
    X: np.ndarray,
    y: np.ndarray,
    thresholds: Dict[str, float] | None = None,
    threshold_proba: float = 0.5,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    metrics_rows: List[Dict[str, float]] = []
    decision_rows: List[Dict[str, float]] = []

    for algo, model in models.items():
        result = evaluate_model(algo, model, X, y, thresholds=thresholds, threshold_proba=threshold_proba)
        row = {"model": algo, **result.metrics, **result.resources}
        metrics_rows.append(row)
        decision_rows.append({"model": algo, **result.decision})

    return metrics_rows, decision_rows
