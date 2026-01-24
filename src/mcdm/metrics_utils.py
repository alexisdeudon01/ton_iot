from __future__ import annotations

import math
from typing import Tuple

INTRINSIC_SCORES = {
    "LR": 1.0,
    "DT": 1.0,
    "RF": 0.0,
    "CNN": 0.0,
    "TabNet": 0.5,
}


def compute_f_perf(f1: float, recall: float, auc: float, gap: float) -> float:
    return 0.4 * f1 + 0.3 * recall + 0.2 * auc - 0.1 * gap


def compute_f_expl(s_intrinsic: float, shap_available: bool, n_params: int, shap_std: float) -> float:
    s_shap = 1.0 if shap_available else 0.0
    s_comp = 1.0 / math.log(1 + max(n_params, 1))
    s_stab = max(0.0, 1.0 - shap_std)
    return 0.4 * s_intrinsic + 0.2 * s_shap + 0.2 * s_comp + 0.2 * s_stab


def compute_f_res(memory_bytes: float, cpu_percent: float, tau_memory: float = 524_288_000, tau_cpu: float = 80.0) -> float:
    m_norm = memory_bytes / tau_memory if tau_memory else 0.0
    cpu_norm = cpu_percent / tau_cpu if tau_cpu else 0.0
    return 0.5 * m_norm + 0.5 * cpu_norm


def explainability_metrics(model_type: str, n_params: int):
    s_intrinsic = INTRINSIC_SCORES.get(model_type, 0.0)
    shap_available = False
    shap_std = 0.5
    return s_intrinsic, shap_available, n_params, shap_std
