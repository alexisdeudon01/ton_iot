from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import plot_tree

from models.dt import DTModel
def _load_processed() -> tuple[pd.DataFrame, list[str]]:
    processed_dir = Path("data/processed")
    cic_path = processed_dir / "cic_processed.csv"
    if not cic_path.exists():
        raise FileNotFoundError("Processed CIC dataset missing. Run preprocessing first.")
    df = pd.read_csv(cic_path)
    feature_order = [c for c in df.columns if c not in ("y", "source_file")]
    return df, feature_order


def _plot_decision_tree(df: pd.DataFrame, feature_order: list[str], out_path: Path) -> None:
    X = df[feature_order].to_numpy()
    y = df["y"].to_numpy()
    model = DTModel(feature_order)
    model.fit(X, y)

    fig, ax = plt.subplots(figsize=(16, 10))
    plot_tree(
        model.model,
        feature_names=feature_order,
        class_names=["Benign", "Attack"],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Decision Tree Structure")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_cnn_architecture(feature_count: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    y = 0.9
    blocks = [
        f"Input ({feature_count} features)",
        "Linear 64 + BatchNorm + ReLU",
        "Dropout 0.2",
        "Linear 32 + ReLU",
        "Linear 2 + Softmax",
    ]
    for block in blocks:
        ax.text(0.5, y, block, ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round", fc="#F4A6A3"))
        y -= 0.15
    ax.set_title("CNN Architecture (Tabular)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_model_figures(graphs_dir: Path) -> None:
    graphs_dir.mkdir(parents=True, exist_ok=True)
    df, feature_order = _load_processed()

    _plot_decision_tree(df, feature_order, graphs_dir / "figure_06_decision_tree.png")
    _plot_cnn_architecture(len(feature_order), graphs_dir / "figure_07_cnn_architecture.png")
