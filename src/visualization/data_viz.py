from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _load_processed() -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    processed_dir = Path("data/processed")
    cic_path = processed_dir / "cic_processed.csv"
    ton_path = processed_dir / "ton_processed.csv"
    if not cic_path.exists() or not ton_path.exists():
        raise FileNotFoundError("Processed datasets not found. Run preprocessing first.")

    cic_df = pd.read_csv(cic_path)
    ton_df = pd.read_csv(ton_path)

    features_path = processed_dir / "common_features.json"
    if features_path.exists():
        features = json.loads(features_path.read_text(encoding="utf-8"))
    else:
        features = [c for c in cic_df.columns if c not in ("y", "source_file")][:15]

    return cic_df, ton_df, features


def _plot_class_distribution(df: pd.DataFrame, title: str, out_path: Path) -> None:
    counts = df["y"].value_counts().sort_index()
    labels = counts.index.tolist()
    values = counts.values.tolist()

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color="#2E86AB")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_ks_results(stats: List[float], pvals: List[float], features: List[str], out_dir: Path) -> None:
    x = np.arange(len(features))

    plt.figure(figsize=(10, 4))
    plt.bar(x, stats, color="#F77F00")
    plt.xticks(x, features, rotation=45, ha="right", fontsize=8)
    plt.title("K-S statistics (CIC vs TON)")
    plt.ylabel("KS statistic")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_03_ks_stats.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(x, pvals, color="#06A77D")
    plt.axhline(0.05, color="red", linestyle="--", linewidth=1)
    plt.xticks(x, features, rotation=45, ha="right", fontsize=8)
    plt.title("K-S p-values (CIC vs TON)")
    plt.ylabel("p-value")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_04_ks_pvalues.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_preprocessing_effect(raw_series: np.ndarray, processed_series: np.ndarray, feature: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.hist(raw_series, bins=50, alpha=0.5, label="Raw", color="#2E86AB")
    plt.hist(processed_series, bins=50, alpha=0.5, label="Processed", color="#F77F00")
    plt.title(f"Preprocessing effect on {feature}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_data_figures(graphs_dir: Path) -> None:
    graphs_dir.mkdir(parents=True, exist_ok=True)

    cic_df, ton_df, features = _load_processed()

    _plot_class_distribution(cic_df, "Class distribution - CIC", graphs_dir / "figure_01_class_distribution_cic.png")
    _plot_class_distribution(ton_df, "Class distribution - TON", graphs_dir / "figure_02_class_distribution_ton.png")

    stats = []
    pvals = []
    for feat in features:
        c = cic_df[feat].dropna().to_numpy()
        t = ton_df[feat].dropna().to_numpy()
        if len(c) == 0 or len(t) == 0:
            stats.append(0.0)
            pvals.append(1.0)
            continue
        res = ks_2samp(c, t, alternative="two-sided", mode="auto")
        stats.append(res.statistic)
        pvals.append(res.pvalue)
    _plot_ks_results(stats, pvals, features, graphs_dir)

    # Preprocessing effect: compare first feature vs log1p transformed
    if features:
        feature = features[0]
        raw_series = cic_df[feature].to_numpy()
        raw_series = raw_series[np.isfinite(raw_series)]
        processed_series = np.log1p(np.clip(raw_series, a_min=0, a_max=None))
        _plot_preprocessing_effect(raw_series, processed_series, feature, graphs_dir / "figure_05_preprocessing_effect.png")
