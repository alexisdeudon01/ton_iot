from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _load_decision_matrix() -> pd.DataFrame:
    path = Path("results") / "decision_matrix.csv"
    if not path.exists():
        raise FileNotFoundError("decision_matrix.csv not found. Run evaluation first.")
    return pd.read_csv(path)


def _load_ranking() -> pd.DataFrame:
    path = Path("results") / "topsis_ranking.csv"
    if not path.exists():
        raise FileNotFoundError("topsis_ranking.csv not found. Run MCDM first.")
    return pd.read_csv(path)


def _pareto_front(matrix: np.ndarray) -> list[int]:
    n = len(matrix)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            better_perf = matrix[j, 0] >= matrix[i, 0]
            better_expl = matrix[j, 1] >= matrix[i, 1]
            better_res = matrix[j, 2] <= matrix[i, 2]
            strictly = matrix[j, 0] > matrix[i, 0] or matrix[j, 1] > matrix[i, 1] or matrix[j, 2] < matrix[i, 2]
            if better_perf and better_expl and better_res and strictly:
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


def generate_mcdm_figures(graphs_dir: Path) -> None:
    graphs_dir.mkdir(parents=True, exist_ok=True)

    decision_df = _load_decision_matrix()
    ranking_df = _load_ranking()

    # Figure 8: Decision matrix heatmap
    heat_df = decision_df.set_index("model")[["f_perf", "f_expl", "f_res"]]
    plt.figure(figsize=(8, 5))
    sns.heatmap(heat_df, annot=True, fmt=".2f", cmap="RdYlGn", linewidths=0.5)
    plt.title("Decision Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(graphs_dir / "figure_08_decision_matrix_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 9: TOPSIS ranking
    plt.figure(figsize=(6, 4))
    plt.bar(ranking_df["model"], ranking_df["closeness"], color="#2E86AB")
    plt.title("TOPSIS Ranking")
    plt.ylabel("Closeness")
    plt.tight_layout()
    plt.savefig(graphs_dir / "figure_09_topsis_ranking.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 10: Radar chart
    labels = ["f_perf", "f_expl", "f_res_inv"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    for _, row in decision_df.iterrows():
        values = [row["f_perf"], row["f_expl"], 1 - row["f_res"]]
        values += values[:1]
        ax.plot(angles, values, label=row["model"])
        ax.fill(angles, values, alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Radar chart")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(graphs_dir / "figure_10_radar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 11: 3D Pareto front
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    matrix = decision_df[["f_perf", "f_expl", "f_res"]].to_numpy()
    pareto_idx = _pareto_front(matrix)
    for i, row in decision_df.iterrows():
        color = "green" if i in pareto_idx else "red"
        ax.scatter(row["f_perf"], row["f_expl"], row["f_res"], c=color, s=80)
        ax.text(row["f_perf"], row["f_expl"], row["f_res"], row["model"])
    ax.set_xlabel("f_perf")
    ax.set_ylabel("f_expl")
    ax.set_zlabel("f_res")
    ax.set_title("3D Pareto Front")
    plt.tight_layout()
    plt.savefig(graphs_dir / "figure_11_pareto_front.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Sensitivity analysis results (winner map / win frequency)
    sens_dir = Path("results") / "sensitivity"
    winners = []
    if sens_dir.exists():
        for csv_path in sorted(sens_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            winners.append(df.iloc[0]["model"])

    if winners:
        models = sorted(set(winners))
        win_counts = {m: winners.count(m) for m in models}

        # Figure 12: Winner map (config vs model)
        matrix = np.zeros((len(winners), len(models)))
        for i, m in enumerate(winners):
            matrix[i, models.index(m)] = 1
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, cmap="Blues", cbar=False)
        plt.yticks([])
        plt.xticks(np.arange(len(models)) + 0.5, models)
        plt.title("Winner Map (Sensitivity)")
        plt.tight_layout()
        plt.savefig(graphs_dir / "figure_12_winner_map.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Figure 13: Win frequency
        plt.figure(figsize=(6, 4))
        plt.bar(win_counts.keys(), win_counts.values(), color="#06A77D")
        plt.title("Win Frequency")
        plt.ylabel("Wins")
        plt.tight_layout()
        plt.savefig(graphs_dir / "figure_13_win_frequency.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        # Placeholder if sensitivity not run
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "Sensitivity results missing", ha="center", va="center")
        plt.axis("off")
        plt.savefig(graphs_dir / "figure_12_winner_map.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "Sensitivity results missing", ha="center", va="center")
        plt.axis("off")
        plt.savefig(graphs_dir / "figure_13_win_frequency.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Figure 14: 3D trade-off space
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    for _, row in decision_df.iterrows():
        ax.scatter(row["f_perf"], row["f_expl"], row["f_res"], s=80)
    ax.set_xlabel("f_perf")
    ax.set_ylabel("f_expl")
    ax.set_zlabel("f_res")
    ax.set_title("3D Trade-off Space")
    plt.tight_layout()
    plt.savefig(graphs_dir / "figure_14_tradeoff_space.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 15: Sensitivity summary (if available)
    plt.figure(figsize=(6, 4))
    if winners:
        series = pd.Series(winners).value_counts()
        plt.bar(series.index, series.values, color="#F77F00")
        plt.title("Sensitivity Summary")
        plt.ylabel("Count")
    else:
        plt.text(0.5, 0.5, "Sensitivity summary missing", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(graphs_dir / "figure_15_sensitivity_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 16-17: 2D projections
    plt.figure(figsize=(6, 4))
    plt.scatter(decision_df["f_perf"], decision_df["f_expl"], c="#2E86AB")
    for _, row in decision_df.iterrows():
        plt.text(row["f_perf"], row["f_expl"], row["model"])
    plt.xlabel("f_perf")
    plt.ylabel("f_expl")
    plt.title("2D Projection: f_perf vs f_expl")
    plt.tight_layout()
    plt.savefig(graphs_dir / "figure_16_projection_perf_expl.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(decision_df["f_perf"], decision_df["f_res"], c="#E63946")
    for _, row in decision_df.iterrows():
        plt.text(row["f_perf"], row["f_res"], row["model"])
    plt.xlabel("f_perf")
    plt.ylabel("f_res")
    plt.title("2D Projection: f_perf vs f_res")
    plt.tight_layout()
    plt.savefig(graphs_dir / "figure_17_projection_perf_res.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 18: Decision guide
    best = ranking_df.iloc[0]["model"] if not ranking_df.empty else "N/A"
    plt.figure(figsize=(6, 4))
    plt.text(0.5, 0.6, f"Recommended Model: {best}", ha="center", fontsize=12)
    plt.text(0.5, 0.4, "Based on AHP-TOPSIS ranking", ha="center", fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(graphs_dir / "figure_18_decision_guide.png", dpi=300, bbox_inches="tight")
    plt.close()
