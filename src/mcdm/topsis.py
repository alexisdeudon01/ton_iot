from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

DEFAULT_CRITERIA_TYPES = {"f_perf": "benefit", "f_expl": "benefit", "f_res": "cost"}


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(matrix, axis=0) + 1e-9
    return matrix / denom


def apply_weights(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return matrix * weights


def topsis_rank(
    decision_df: pd.DataFrame,
    weights: np.ndarray,
    criteria_types: Dict[str, str] | None = None,
) -> pd.DataFrame:
    criteria_types = criteria_types or DEFAULT_CRITERIA_TYPES
    criteria = [c for c in decision_df.columns if c != "model"]

    matrix = decision_df[criteria].to_numpy(dtype=float)
    norm = normalize_matrix(matrix)
    weighted = apply_weights(norm, weights)

    ideal = []
    anti = []
    for idx, crit in enumerate(criteria):
        if criteria_types.get(crit, "benefit") == "cost":
            ideal.append(weighted[:, idx].min())
            anti.append(weighted[:, idx].max())
        else:
            ideal.append(weighted[:, idx].max())
            anti.append(weighted[:, idx].min())
    ideal = np.array(ideal)
    anti = np.array(anti)

    d_pos = np.linalg.norm(weighted - ideal, axis=1)
    d_neg = np.linalg.norm(weighted - anti, axis=1)
    closeness = d_neg / (d_pos + d_neg + 1e-9)

    ranked = pd.DataFrame({"model": decision_df["model"], "closeness": closeness})
    ranked = ranked.sort_values("closeness", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked
