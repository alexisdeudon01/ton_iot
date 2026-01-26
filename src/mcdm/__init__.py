from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.mcdm.ahp import resolve_ahp_weights
from src.mcdm.topsis import topsis_rank
from src.mcdm.sensitivity import run_sensitivity_analysis as _run_sensitivity_analysis


def run_ahp_topsis(config: Dict) -> Path:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    decision_path = results_dir / "decision_matrix.csv"
    if not decision_path.exists():
        raise FileNotFoundError("decision_matrix.csv not found. Run evaluation first.")

    decision_df = pd.read_csv(decision_path)
    criteria = [c for c in decision_df.columns if c != "model"]
    weights = resolve_ahp_weights(config, criteria)
    ranking = topsis_rank(decision_df, weights)
    ranking_path = results_dir / "topsis_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    return ranking_path


def run_sensitivity_analysis(config: Dict) -> Path:
    results_dir = Path("results")
    decision_path = results_dir / "decision_matrix.csv"
    if not decision_path.exists():
        raise FileNotFoundError("decision_matrix.csv not found. Run evaluation first.")

    decision_df = pd.read_csv(decision_path)
    criteria = [c for c in decision_df.columns if c != "model"]
    sensitivity_results = _run_sensitivity_analysis(decision_df, criteria)

    out_dir = results_dir / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in sensitivity_results.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return out_dir


__all__ = ["run_ahp_topsis", "run_sensitivity_analysis"]
