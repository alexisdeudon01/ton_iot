from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.mcdm.topsis import topsis_rank


def run_sensitivity(
    decision_df: pd.DataFrame,
    profiles: Dict[str, List[float]],
    criteria: List[str],
) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    for name, weights in profiles.items():
        w = np.array(weights, dtype=float)
        if w.sum() == 0:
            w = np.array([1.0 / len(criteria)] * len(criteria), dtype=float)
        else:
            w = w / w.sum()
        ranking = topsis_rank(decision_df, w)
        results[name] = ranking
    return results


def _generate_weight_configs(step: float = 0.1, limit: int = 66) -> List[List[float]]:
    weights: List[List[float]] = []
    steps = int(round(1 / step))
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            k = steps - i - j
            w = [i * step, j * step, k * step]
            weights.append(w)
    weights = sorted(weights, key=lambda w: (w[0], w[1], w[2]))
    return weights[:limit]


def run_sensitivity_analysis(decision_df: pd.DataFrame, criteria: List[str]) -> Dict[str, pd.DataFrame]:
    profiles = _generate_weight_configs(step=0.1, limit=66)
    results: Dict[str, pd.DataFrame] = {}
    for idx, weights in enumerate(profiles, start=1):
        w = np.array(weights, dtype=float)
        w = w / (w.sum() or 1.0)
        ranking = topsis_rank(decision_df, w)
        results[f"cfg_{idx:02d}"] = ranking
    return results
