from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from evaluation.performance import PerformanceEvaluator
from evaluation.resources import ResourceTracker
from src.mcdm.objectives import compute_f_expl, compute_f_perf, compute_f_res, explainability_metrics


@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    objectives: Dict[str, float]
    resources: Dict[str, float]


def _predict_labels(probas: np.ndarray, threshold: float) -> np.ndarray:
    if probas.ndim == 1:
        return (probas >= threshold).astype(int)
    return (probas[:, 1] >= threshold).astype(int)


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


def evaluate_model(model, algo_key: str, X: np.ndarray, y: np.ndarray, thresholds: Dict[str, float]) -> EvaluationResult:
    tracker = ResourceTracker()
    tracker.start()
    probas = model.predict_proba(X)
    usage = tracker.get_usage()

    y_pred = _predict_labels(probas, threshold=0.5)
    metrics = PerformanceEvaluator.compute_metrics(y, y_pred)

    try:
        auc = roc_auc_score(y, probas[:, 1])
    except Exception:
        auc = 0.0
    metrics["roc_auc"] = float(auc)

    n_params = _estimate_n_params(model.model if hasattr(model, "model") else model)
    s_intrinsic, shap_available, n_params, shap_std = explainability_metrics(algo_key, n_params)

    f_perf = compute_f_perf(metrics["f1"], metrics["recall"], float(auc), gap=0.0)
    f_expl = compute_f_expl(s_intrinsic, shap_available, n_params, shap_std)

    mem_bytes = usage["ram_mb"] * 1024 * 1024
    tau_memory = thresholds.get("tau_memory", thresholds.get("memory_mb", 500)) * 1024 * 1024
    tau_cpu = thresholds.get("tau_cpu", thresholds.get("cpu_percent", 80.0))
    f_res = compute_f_res(mem_bytes, usage["cpu_percent"], tau_memory, tau_cpu)

    objectives = {"f_perf": float(f_perf), "f_expl": float(f_expl), "f_res": float(f_res)}
    return EvaluationResult(metrics=metrics, objectives=objectives, resources=usage)


def build_results_frames(results: Dict[str, EvaluationResult]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows = []
    decision_rows = []
    for algo, res in results.items():
        metrics_rows.append({"model": algo, **res.metrics, **res.resources})
        decision_rows.append({"model": algo, **res.objectives})
    return pd.DataFrame(metrics_rows), pd.DataFrame(decision_rows)
