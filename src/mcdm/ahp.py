from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def normalize_weights(weights: Dict[str, float], criteria: Iterable[str]) -> np.ndarray:
    values = np.array([weights.get(c, 0.0) for c in criteria], dtype=float)
    total = values.sum()
    if total == 0:
        return np.array([1.0 / len(values)] * len(values), dtype=float)
    return values / total


def compute_ahp_weights_from_pairwise(pairwise: List[List[float]]) -> np.ndarray:
    matrix = np.array(pairwise, dtype=float)
    vals, vecs = np.linalg.eig(matrix)
    idx = np.argmax(vals.real)
    principal = vecs[:, idx].real
    principal = principal / principal.sum()
    return principal


def resolve_ahp_weights(config: Dict, criteria: List[str]) -> np.ndarray:
    if not config:
        return np.array([1.0 / len(criteria)] * len(criteria), dtype=float)

    if "ahp_weights" in config:
        weights_cfg = config.get("ahp_weights", {})
        mapped = {
            "f_perf": weights_cfg.get("w_perf", 0.33),
            "f_expl": weights_cfg.get("w_expl", 0.33),
            "f_res": weights_cfg.get("w_res", 0.33),
        }
        return normalize_weights(mapped, criteria)

    ahp_cfg = config.get("ahp", {})
    if "pairwise" in ahp_cfg:
        return compute_ahp_weights_from_pairwise(ahp_cfg["pairwise"])
    if "weights" in ahp_cfg:
        return normalize_weights(ahp_cfg["weights"], criteria)

    # Fallback to explicit defaults (f_perf, f_expl, f_res)
    defaults = {"f_perf": 0.50, "f_expl": 0.30, "f_res": 0.20}
    return normalize_weights(defaults, criteria)
