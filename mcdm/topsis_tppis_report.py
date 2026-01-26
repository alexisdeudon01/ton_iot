import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from src.mcdm.metrics_utils import compute_f_perf, compute_f_expl, compute_f_res

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

CRITERIA_COLUMNS = ["f_perf", "f_expl", "f_res"]
CRITERIA_TYPES = {"f_perf": "benefit", "f_expl": "benefit", "f_res": "cost"}


def build_df_sources_from_run_report(run_report: Dict) -> pd.DataFrame:
    rows = []
    for model_name, metrics in run_report.items():
        if model_name == "_metadata":
            continue
        if not model_name.startswith("fused_") or model_name == "fused_global":
            continue
        mcdm_inputs = metrics.get("mcdm_inputs", {})
        shap_std = mcdm_inputs.get("shap_std", 0.5)
        n_params_raw = mcdm_inputs.get("n_params") or metrics.get("complexity") or metrics.get("n_params")
        try:
            n_params = int(n_params_raw) if n_params_raw is not None else 1
        except Exception:
            n_params = 1
        n_params = max(n_params, 1)
        mem_bytes = float(mcdm_inputs.get("memory_bytes", 0.0))
        cpu_percent = float(mcdm_inputs.get("cpu_percent", metrics.get("cpu_percent", 0.0)))
        gap = float(metrics.get("gap", 0.0))
        f1 = float(metrics.get("f1", 0.0))
        recall = float(metrics.get("recall", 0.0))
        auc = float(metrics.get("roc_auc", 0.0))
        faithfulness = float(metrics.get("faithfulness", mcdm_inputs.get("s_intrinsic", 0.0)))
        shap_available = bool(mcdm_inputs.get("shap_available", False))
        f_perf = compute_f_perf(f1, recall, auc, gap)
        f_expl = compute_f_expl(faithfulness, shap_available, n_params, shap_std)
        f_res = compute_f_res(mem_bytes, cpu_percent)
        rows.append({
            "model": model_name.replace("fused_", ""),
            "f1": f1,
            "recall": recall,
            "roc_auc": auc,
            "gap": gap,
            "faithfulness": faithfulness,
            "shap_available": shap_available,
            "n_params": n_params,
            "memory_bytes": mem_bytes,
            "cpu_percent": cpu_percent,
            "f_perf": f_perf,
            "f_expl": f_expl,
            "f_res": f_res,
        })
    return pd.DataFrame(rows)


def normalize_matrix(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    norm = df.copy()
    for col in columns:
        values = norm[col].to_numpy(dtype=float)
        v_min = np.min(values)
        v_max = np.max(values)
        if v_max - v_min == 0:
            norm[col] = 0.0
            continue
        if CRITERIA_TYPES.get(col) == "cost":
            norm[col] = (v_max - values) / (v_max - v_min)
        else:
            norm[col] = (values - v_min) / (v_max - v_min)
    return norm


def ks_validation_stats(df_features: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    stats_rows = []
    clean = df_features[features].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(clean)
    if n == 0:
        return pd.DataFrame(columns=["feature", "ks_stat", "critical_value"])
    critical_value = 1.36 / np.sqrt(n)
    for feat in features:
        series = clean[feat].astype(float)
        if series.std() == 0:
            ks_stat = 0.0
        else:
            standardized = (series - series.mean()) / (series.std() + 1e-9)
            ks_stat = stats.kstest(standardized, "norm")[0]
        stats_rows.append({"feature": feat, "ks_stat": ks_stat, "critical_value": critical_value})
    return pd.DataFrame(stats_rows)


def compute_feature_importance(df_features: pd.DataFrame, y: pd.Series, features: List[str]) -> pd.DataFrame:
    X = df_features[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_clean = y.loc[X.index]
    try:
        from sklearn.feature_selection import mutual_info_classif
        importances = mutual_info_classif(X.values, y_clean.values, discrete_features=False, random_state=42)
    except Exception:
        importances = X.var().to_numpy()
    return pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False)


def plot_ks_validation(ks_df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(12, 5))
    if ks_df.empty:
        plt.text(0.5, 0.5, "No KS data available", ha='center', va='center')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    colors = ["red" if row.ks_stat > row.critical_value else "blue" for row in ks_df.itertuples()]
    plt.scatter(ks_df["feature"], ks_df["ks_stat"], c=colors, s=80)
    plt.axhline(ks_df["critical_value"].iloc[0], color="black", linestyle="--", label="Critical value")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("KS statistic")
    plt.title("KS validation: features vs theoretical threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(fi_df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=fi_df, x="importance", y="feature", color="#3498db")
    plt.title("Feature importance (15 universal features)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_split_pie(train_ratio: float, val_ratio: float, test_ratio: float, output_path: str) -> None:
    labels = ["Train", "Validation", "Test"]
    sizes = [train_ratio, val_ratio, test_ratio]
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Data split proportions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_heatmap(df_matrix: pd.DataFrame, title: str, output_path: str) -> None:
    plt.figure(figsize=(8, 5))
    sns.heatmap(df_matrix, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_ahp_weights(weights: List[float]) -> np.ndarray:
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    pairwise = w[:, None] / w[None, :]
    vals, vecs = np.linalg.eig(pairwise)
    idx = np.argmax(vals.real)
    principal = vecs[:, idx].real
    principal = principal / principal.sum()
    return principal


def plot_ahp_weights(weights: np.ndarray, labels: List[str], output_path: str) -> None:
    plt.figure(figsize=(6, 6))
    wedges, texts = plt.pie(weights, labels=labels, startangle=90, wedgeprops=dict(width=0.4))
    plt.title("AHP weights")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def topsis_rank(df_matrix: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    matrix = df_matrix.to_numpy(dtype=float)
    norm = matrix / (np.linalg.norm(matrix, axis=0) + 1e-9)
    weighted = norm * weights
    # benefit for first two, cost for last
    ideal = np.array([
        weighted[:, 0].max(),
        weighted[:, 1].max(),
        weighted[:, 2].min(),
    ])
    anti = np.array([
        weighted[:, 0].min(),
        weighted[:, 1].min(),
        weighted[:, 2].max(),
    ])
    d_pos = np.linalg.norm(weighted - ideal, axis=1)
    d_neg = np.linalg.norm(weighted - anti, axis=1)
    cc = d_neg / (d_pos + d_neg + 1e-9)
    return pd.DataFrame({"model": df_matrix.index, "closeness": cc}).sort_values("closeness", ascending=False)


def plot_topsis_closeness(df_rank: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(10, 6))
    models = df_rank["model"].tolist()
    cc = df_rank["closeness"].to_numpy()
    plt.bar(models, cc, color="#2ecc71", label="Closeness")
    plt.bar(models, 1 - cc, bottom=cc, color="#ecf0f1", label="1 - Closeness")
    plt.ylabel("Closeness coefficient (C*)")
    plt.title("TOPSIS closeness coefficients")
    plt.legend()
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_top3_radar(df_norm: pd.DataFrame, top3: List[str], output_path: str) -> None:
    categories = ["Performance", "Explainability", "Resources (inv)"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for model in top3:
        row = df_norm.loc[model]
        values = [row["f_perf"], row["f_expl"], row["f_res"]]
        values += values[:1]
        ax.plot(angles, values, label=model)
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title("Top-3 alternatives (radar)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pipeline_overview(output_path: str) -> None:
    steps = [
        "Data validation",
        "Feature importance",
        "Data split",
        "Decision matrix",
        "Normalization",
        "AHP weighting",
        "TOPSIS ranking",
        "Final comparison",
    ]
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")
    x_positions = np.linspace(0.05, 0.95, len(steps))
    y = 0.5
    for i, (x, label) in enumerate(zip(x_positions, steps)):
        ax.add_patch(plt.Rectangle((x - 0.055, y - 0.12), 0.11, 0.24, fill=False, linewidth=1.5))
        ax.text(x, y, label, ha="center", va="center", fontsize=9)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x_positions[i + 1] - 0.06, y), xytext=(x + 0.06, y),
                        arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_title("MCDA pipeline overview", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def generate_topsis_report(
    df_sources: pd.DataFrame,
    df_features: pd.DataFrame,
    y: pd.Series,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    weights: List[float] = None,
) -> Dict[str, List[str]]:
    os.makedirs(output_dir, exist_ok=True)

    # Ensure criteria columns
    if not all(col in df_sources.columns for col in CRITERIA_COLUMNS):
        raise ValueError("df_sources must contain f_perf, f_expl, f_res columns")

    feature_cols = [c for c in df_features.columns if c != "y"]
    ks_df = ks_validation_stats(df_features, feature_cols)
    fi_df = compute_feature_importance(df_features, y, feature_cols)

    # Plots
    outputs = []
    ks_path = os.path.join(output_dir, "01_ks_validation.png")
    plot_ks_validation(ks_df, ks_path)
    outputs.append(ks_path)

    fi_path = os.path.join(output_dir, "02_feature_importance.png")
    plot_feature_importance(fi_df, fi_path)
    outputs.append(fi_path)

    split_path = os.path.join(output_dir, "03_data_split_pie.png")
    plot_split_pie(train_ratio, val_ratio, test_ratio, split_path)
    outputs.append(split_path)

    raw_matrix = df_sources.set_index("model")[CRITERIA_COLUMNS]
    raw_path = os.path.join(output_dir, "04_raw_matrix_heatmap.png")
    plot_heatmap(raw_matrix, "Raw decision matrix", raw_path)
    outputs.append(raw_path)

    norm_matrix = normalize_matrix(raw_matrix, CRITERIA_COLUMNS)
    norm_path = os.path.join(output_dir, "05_normalized_matrix_heatmap.png")
    plot_heatmap(norm_matrix, "Normalized decision matrix", norm_path)
    outputs.append(norm_path)

    # AHP weights from pillar weights (performance, explainability, resources)
    if weights is None:
        weights = [0.70, 0.15, 0.15]
    ahp_weights = compute_ahp_weights(weights)
    ahp_path = os.path.join(output_dir, "06_ahp_weights_donut.png")
    plot_ahp_weights(ahp_weights, ["Performance", "Explainability", "Resources"], ahp_path)
    outputs.append(ahp_path)

    # TOPSIS ranking
    topsis_ranked = topsis_rank(norm_matrix, ahp_weights)
    topsis_path = os.path.join(output_dir, "07_topsis_closeness.png")
    plot_topsis_closeness(topsis_ranked, topsis_path)
    outputs.append(topsis_path)

    # Radar top 3
    top3 = topsis_ranked["model"].head(3).tolist()
    radar_path = os.path.join(output_dir, "08_top3_radar.png")
    plot_top3_radar(norm_matrix, top3, radar_path)
    outputs.append(radar_path)

    pipeline_path = os.path.join(output_dir, "09_pipeline_overview.png")
    plot_pipeline_overview(pipeline_path)
    outputs.append(pipeline_path)

    # Save matrices
    raw_matrix.to_csv(os.path.join(output_dir, "raw_decision_matrix.csv"))
    norm_matrix.to_csv(os.path.join(output_dir, "normalized_decision_matrix.csv"))
    topsis_ranked.to_csv(os.path.join(output_dir, "topsis_ranking.csv"), index=False)

    return {"plots": outputs}
