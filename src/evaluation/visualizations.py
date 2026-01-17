#!/usr/bin/env python3
"""
Generate all Phase 3 visualizations (27 PNG files)
Matplotlib only (no seaborn)
Handles missing SHAP/LIME gracefully
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.debug("SHAP not available - explainability visualizations may be limited")

try:
    import lime
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.debug("LIME not available - explainability visualizations may be limited")


def save_fig(fig, filepath: Path):
    """Save figure to PNG"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.debug(f"Saved: {filepath}")


def generate_all_visualizations(
    metrics_df: pd.DataFrame,
    metrics_by_fold_df: Optional[pd.DataFrame] = None,
    scores_normalized_df: Optional[pd.DataFrame] = None,
    confusion_matrices: Optional[Dict[str, np.ndarray]] = None,
    roc_curves: Optional[Dict[str, Tuple[np.ndarray, np.ndarray, float]]] = None,
    pr_curves: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    shap_values_dict: Optional[Dict[str, np.ndarray]] = None,
    lime_importances_dict: Optional[Dict[str, np.ndarray]] = None,
    feature_names: Optional[List[str]] = None,
    output_dir: Path = Path("output/phase3_evaluation")
) -> Dict[str, Path]:
    """
    Generate all 27 Phase 3 visualizations

    Args:
        metrics_df: Aggregated metrics DataFrame (algo, f1_mean, precision_mean, etc.)
        metrics_by_fold_df: Per-fold metrics (algo, fold, f1, precision, etc.)
        scores_normalized_df: Normalized scores (algo, perf_score, resource_score, explain_score)
        confusion_matrices: Dict mapping algo name to confusion matrix (n_classes, n_classes)
        roc_curves: Dict mapping algo name to (fpr, tpr, auc)
        pr_curves: Dict mapping algo name to (precision, recall)
        shap_values_dict: Dict mapping algo name to SHAP values (n_samples, n_features)
        lime_importances_dict: Dict mapping algo name to LIME importances (n_features,)
        feature_names: List of feature names (for SHAP/LIME)
        output_dir: Output directory for PNG files

    Returns:
        Dictionary mapping visualization names to file paths
    """
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    generated_files = {}

    # DIM 1: PERFORMANCE (7 visualizations)
    logger.info("Generating Performance visualizations...")
    generated_files.update(generate_performance_visualizations(
        metrics_df, metrics_by_fold_df, confusion_matrices, roc_curves, pr_curves, vis_dir
    ))

    # DIM 2: RESOURCES (7 visualizations)
    logger.info("Generating Resource visualizations...")
    generated_files.update(generate_resource_visualizations(
        metrics_df, scores_normalized_df, vis_dir
    ))

    # DIM 3: EXPLAINABILITY (6 visualizations)
    logger.info("Generating Explainability visualizations...")
    generated_files.update(generate_explainability_visualizations(
        metrics_df, scores_normalized_df, shap_values_dict, lime_importances_dict,
        feature_names, vis_dir
    ))

    # TRANSVERSAL 3D (7 visualizations)
    logger.info("Generating 3D transversal visualizations...")
    generated_files.update(generate_3d_transversal_visualizations(
        scores_normalized_df, metrics_df, vis_dir
    ))

    # Generate INDEX.md
    generate_visualizations_index(generated_files, vis_dir)

    logger.info(f"✅ Generated {len(generated_files)} visualizations in {vis_dir}")
    return generated_files


# ============================================================================
# DIM 1: PERFORMANCE VISUALIZATIONS (7)
# ============================================================================
def get_algo_names(df: pd.DataFrame) -> pd.Series:
    """
    Return algorithm names in a consistent way.
    Enforces 'algo' as a first-class column.
    """
    if 'algo' in df.columns:
        return df['algo'].astype(str)

    if df.index.name == 'algo':
        return df.index.astype(str).to_series(index=df.index)

    raise ValueError(
        "Algorithm names not found. Expected 'algo' column or index named 'algo'."
    )

def generate_performance_visualizations(
    metrics_df: pd.DataFrame,
    metrics_by_fold_df: Optional[pd.DataFrame],
    confusion_matrices: Optional[Dict[str, np.ndarray]],
    roc_curves: Optional[Dict[str, Tuple[np.ndarray, np.ndarray, float]]],
    pr_curves: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    vis_dir: Path
) -> Dict[str, Path]:
    """Generate all performance dimension visualizations"""
    generated = {}

    # 1. perf_f1_bar.png
    fig, ax = plt.subplots(figsize=(10, 6))
    algos = metrics_df['algo'] if 'algo' in metrics_df.columns else metrics_df.index
    f1_means = metrics_df['f1_mean'] if 'f1_mean' in metrics_df.columns else metrics_df.get('f1_score', [])
    f1_stds = metrics_df.get('f1_std', [0] * len(f1_means))

    bars = ax.bar(algos, f1_means, yerr=f1_stds, capsize=5, color='steelblue', alpha=0.7)
    ax.set_title('F1 Score by Algorithm (Mean ± Std)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    for bar, mean, std in zip(bars, f1_means, f1_stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    path = vis_dir / "perf_f1_bar.png"
    save_fig(fig, path)
    generated['perf_f1_bar'] = path

    # 2. perf_metrics_grouped_bar.png
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(algos))
    width = 0.2
    metrics_cols = ['f1_mean', 'precision_mean', 'recall_mean', 'accuracy']
    metric_labels = ['F1', 'Precision', 'Recall', 'Accuracy']
    colors = ['steelblue', 'coral', 'lightgreen', 'gold']

    for i, (col, label, color) in enumerate(zip(metrics_cols, metric_labels, colors)):
        if col in metrics_df.columns:
            values = metrics_df[col]
            ax.bar(x + i*width, values, width, label=label, color=color, alpha=0.7)

    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Metrics by Algorithm', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(algos, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    path = vis_dir / "perf_metrics_grouped_bar.png"
    save_fig(fig, path)
    generated['perf_metrics_grouped_bar'] = path

    # 3. perf_f1_boxplot.png
    if metrics_by_fold_df is not None and 'f1' in metrics_by_fold_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        algos_unique = metrics_by_fold_df['algo'].unique()
        data = [metrics_by_fold_df[metrics_by_fold_df['algo'] == algo]['f1'].values for algo in algos_unique]
        bp = ax.boxplot(data, patch_artist=True)
        ax.set_xticklabels(algos_unique)
        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.7)
        ax.set_title('F1 Score Distribution by Fold (Boxplot)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        path = vis_dir / "perf_f1_boxplot.png"
        save_fig(fig, path)
        generated['perf_f1_boxplot'] = path

    # 4-6. Confusion matrices, ROC, PR curves per algo
    if confusion_matrices:
        for algo, cm in confusion_matrices.items():
            # Confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.figure.colorbar(im, ax=ax)
            classes = range(cm.shape[0])
            ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
                   xticklabels=classes, yticklabels=classes,
                   title=f'Confusion Matrix - {algo}', ylabel='True Label', xlabel='Predicted Label')
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
            path = vis_dir / f"perf_confusion_matrix_{algo}.png"
            save_fig(fig, path)
            generated[f'perf_confusion_matrix_{algo}'] = path

    if roc_curves:
        for algo, (fpr, tpr, auc) in roc_curves.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, lw=2, label=f'{algo} (AUC = {auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'ROC Curve - {algo}', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            path = vis_dir / f"perf_roc_{algo}.png"
            save_fig(fig, path)
            generated[f'perf_roc_{algo}'] = path

    if pr_curves:
        for algo, (precision, recall) in pr_curves.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recall, precision, lw=2, label=algo)
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_title(f'Precision-Recall Curve - {algo}', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            path = vis_dir / f"perf_pr_{algo}.png"
            save_fig(fig, path)
            generated[f'perf_pr_{algo}'] = path

    # 7. perf_metrics_heatmap.png
    fig, ax = plt.subplots(figsize=(10, 6))
    perf_cols = ['f1_mean', 'precision_mean', 'recall_mean', 'accuracy']
    heatmap_data = metrics_df[perf_cols].T if all(c in metrics_df.columns for c in perf_cols) else pd.DataFrame()
    if not heatmap_data.empty:
        im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(np.arange(len(algos)))
        ax.set_yticks(np.arange(len(perf_cols)))
        ax.set_xticklabels(algos, rotation=45, ha='right')
        ax.set_yticklabels([c.replace('_mean', '').title() for c in perf_cols])
        ax.set_title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')
        for i in range(len(perf_cols)):
            for j in range(len(algos)):
                text = ax.text(j, i, f'{heatmap_data.values[i, j]:.3f}', ha="center", va="center", color="black")
        plt.colorbar(im, ax=ax)
    path = vis_dir / "perf_metrics_heatmap.png"
    save_fig(fig, path)
    generated['perf_metrics_heatmap'] = path

    return generated


# ============================================================================
# DIM 2: RESOURCE VISUALIZATIONS (7)
# ============================================================================

def generate_resource_visualizations(
    metrics_df: pd.DataFrame,
    scores_normalized_df: Optional[pd.DataFrame],
    vis_dir: Path
) -> Dict[str, Path]:
    """Generate all resource dimension visualizations"""
    generated = {}
    algos = metrics_df['algo'] if 'algo' in metrics_df.columns else metrics_df.index

    # 1. res_train_time_bar.png
    fig, ax = plt.subplots(figsize=(10, 6))
    times = metrics_df.get('train_time_sec', metrics_df.get('training_time_seconds', []))
    time_stds = metrics_df.get('train_time_std', [0] * len(times))
    bars = ax.bar(algos, times, yerr=time_stds, capsize=5, color='coral', alpha=0.7)
    ax.set_title('Training Time by Algorithm (Mean ± Std)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    path = vis_dir / "res_train_time_bar.png"
    save_fig(fig, path)
    generated['res_train_time_bar'] = path

    # 2. res_peak_ram_bar.png
    fig, ax = plt.subplots(figsize=(10, 6))
    rams = metrics_df.get('peak_ram_mb', metrics_df.get('memory_used_mb', []))
    ram_stds = metrics_df.get('peak_ram_std', [0] * len(rams))
    bars = ax.bar(algos, rams, yerr=ram_stds, capsize=5, color='lightgreen', alpha=0.7)
    ax.set_title('Peak RAM Usage by Algorithm (Mean ± Std)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Peak RAM (MB)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    path = vis_dir / "res_peak_ram_bar.png"
    save_fig(fig, path)
    generated['res_peak_ram_bar'] = path

    # 3. res_latency_bar.png
    fig, ax = plt.subplots(figsize=(10, 6))
    latencies = metrics_df.get('latency_ms', [])
    latency_stds = metrics_df.get('latency_std', [0] * len(latencies))
    if len(latencies) > 0:
        bars = ax.bar(algos, latencies, yerr=latency_stds, capsize=5, color='gold', alpha=0.7)
        ax.set_title('Inference Latency by Algorithm (Mean ± Std)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
    path = vis_dir / "res_latency_bar.png"
    save_fig(fig, path)
    generated['res_latency_bar'] = path

    # 4-5. Tradeoff scatter plots
    f1_scores = metrics_df.get('f1_mean', metrics_df.get('f1_score', []))
    if len(f1_scores) > 0 and len(times) > 0:
        # F1 vs Time
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(times, f1_scores, s=100, alpha=0.6, c='steelblue')
        for algo, time, f1 in zip(algos, times, f1_scores):
            ax.annotate(algo, (time, f1), fontsize=9)
        ax.set_xlabel('Training Time (seconds)', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Tradeoff: F1 Score vs Training Time', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        path = vis_dir / "res_tradeoff_f1_vs_time.png"
        save_fig(fig, path)
        generated['res_tradeoff_f1_vs_time'] = path

        # F1 vs RAM
        if len(rams) > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(rams, f1_scores, s=100, alpha=0.6, c='coral')
            for algo, ram, f1 in zip(algos, rams, f1_scores):
                ax.annotate(algo, (ram, f1), fontsize=9)
            ax.set_xlabel('Peak RAM (MB)', fontsize=12)
            ax.set_ylabel('F1 Score', fontsize=12)
            ax.set_title('Tradeoff: F1 Score vs Peak RAM', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            path = vis_dir / "res_tradeoff_f1_vs_ram.png"
            save_fig(fig, path)
            generated['res_tradeoff_f1_vs_ram'] = path

    # 6. res_pareto_frontier.png
    if len(f1_scores) > 0 and len(times) > 0 and len(rams) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Normalize for Pareto (higher is better for all)
        # For time/ram: invert (1/time, 1/ram) or use negative
        norm_times = 1 / (np.array(times) + 1e-10)
        norm_rams = 1 / (np.array(rams) + 1e-10)
        # Combine: composite score = f1 * norm_time * norm_ram
        composite = np.array(f1_scores) * norm_times * norm_rams
        # Find non-dominated points
        dominated = np.zeros(len(algos), dtype=bool)
        for i in range(len(algos)):
            for j in range(len(algos)):
                if i != j and f1_scores[j] >= f1_scores[i] and times[j] <= times[i] and rams[j] <= rams[i]:
                    if f1_scores[j] > f1_scores[i] or times[j] < times[i] or rams[j] < rams[i]:
                        dominated[i] = True
                        break
        ax.scatter(times, f1_scores, s=100, alpha=0.3, c='gray', label='Dominated')
        if not all(dominated):
            ax.scatter(np.array(times)[~dominated], np.array(f1_scores)[~dominated],
                      s=150, alpha=0.8, c='red', marker='*', label='Pareto Frontier')
        for algo, time, f1 in zip(algos, times, f1_scores):
            ax.annotate(algo, (time, f1), fontsize=9)
        ax.set_xlabel('Training Time (seconds)', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Pareto Frontier: F1 vs Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        path = vis_dir / "res_pareto_frontier.png"
        save_fig(fig, path)
        generated['res_pareto_frontier'] = path

    # 7. res_components_heatmap.png
    fig, ax = plt.subplots(figsize=(10, 6))
    res_cols = ['train_time_sec', 'peak_ram_mb', 'latency_ms']
    res_labels = ['Training Time (s)', 'Peak RAM (MB)', 'Latency (ms)']
    available_cols = [c for c in res_cols if c in metrics_df.columns]
    if len(available_cols) > 0:
        heatmap_data = metrics_df[available_cols].T
        im = ax.imshow(heatmap_data.values, cmap='Reds', aspect='auto')
        ax.set_xticks(np.arange(len(algos)))
        ax.set_yticks(np.arange(len(available_cols)))
        ax.set_xticklabels(algos, rotation=45, ha='right')
        ax.set_yticklabels([l for l, c in zip(res_labels, res_cols) if c in available_cols])
        ax.set_title('Resource Components Heatmap', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax)
    path = vis_dir / "res_components_heatmap.png"
    save_fig(fig, path)
    generated['res_components_heatmap'] = path

    return generated


# ============================================================================
# DIM 3: EXPLAINABILITY VISUALIZATIONS (6)
# ============================================================================

def generate_explainability_visualizations(
    metrics_df: pd.DataFrame,
    scores_normalized_df: Optional[pd.DataFrame],
    shap_values_dict: Optional[Dict[str, np.ndarray]],
    lime_importances_dict: Optional[Dict[str, np.ndarray]],
    feature_names: Optional[List[str]],
    vis_dir: Path
) -> Dict[str, Path]:
    """Generate all explainability dimension visualizations"""
    generated = {}
    algos = metrics_df['algo'] if 'algo' in metrics_df.columns else metrics_df.index

    # 1. exp_score_bar.png
    fig, ax = plt.subplots(figsize=(10, 6))
    exp_scores = metrics_df.get('explain_score', metrics_df.get('explainability_score', []))
    if len(exp_scores) > 0:
        bars = ax.bar(algos, exp_scores, color='purple', alpha=0.7)
        ax.set_title('Explainability Score by Algorithm', fontsize=14, fontweight='bold')
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Explainability Score', fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        for bar, score in zip(bars, exp_scores):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    path = vis_dir / "exp_score_bar.png"
    save_fig(fig, path)
    generated['exp_score_bar'] = path

    # 2. exp_components_stacked_bar.png
    fig, ax = plt.subplots(figsize=(12, 6))
    native = metrics_df.get('native_interpretability', [0] * len(algos))
    shap_scores = metrics_df.get('shap_score', [0] * len(algos))
    lime_scores = metrics_df.get('lime_score', [0] * len(algos))
    x = np.arange(len(algos))
    width = 0.6
    bottom = np.zeros(len(algos))
    for comp, label, color in zip([native, shap_scores, lime_scores],
                                   ['Native', 'SHAP', 'LIME'],
                                   ['steelblue', 'coral', 'lightgreen']):
        # Ensure v is not None before comparison
        if any(v is not None and v > 0 for v in comp):
            ax.bar(x, comp, width, bottom=bottom, label=label, color=color, alpha=0.7)
            bottom += np.array(comp)
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Explainability Components (Stacked)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    path = vis_dir / "exp_components_stacked_bar.png"
    save_fig(fig, path)
    generated['exp_components_stacked_bar'] = path

    # 3. exp_tradeoff_f1_vs_explain.png
    f1_scores = metrics_df.get('f1_mean', metrics_df.get('f1_score', []))
    if len(f1_scores) > 0 and len(exp_scores) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(exp_scores, f1_scores, s=100, alpha=0.6, c='purple')
        for algo, exp, f1 in zip(algos, exp_scores, f1_scores):
            ax.annotate(algo, (exp, f1), fontsize=9)
        ax.set_xlabel('Explainability Score', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Tradeoff: F1 Score vs Explainability', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        path = vis_dir / "exp_tradeoff_f1_vs_explain.png"
        save_fig(fig, path)
        generated['exp_tradeoff_f1_vs_explain'] = path

    # 4. exp_components_heatmap.png
    fig, ax = plt.subplots(figsize=(10, 6))
    exp_components = ['native_interpretability', 'shap_score', 'lime_score']
    available = [c for c in exp_components if c in metrics_df.columns]
    if len(available) > 0:
        heatmap_data = metrics_df[available].T
        im = ax.imshow(heatmap_data.values, cmap='Purples', aspect='auto')
        ax.set_xticks(np.arange(len(algos)))
        ax.set_yticks(np.arange(len(available)))
        ax.set_xticklabels(algos, rotation=45, ha='right')
        ax.set_yticklabels([c.replace('_score', '').replace('_', ' ').title() for c in available])
        ax.set_title('Explainability Components Heatmap', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax)
    path = vis_dir / "exp_components_heatmap.png"
    save_fig(fig, path)
    generated['exp_components_heatmap'] = path

    # 5-6. SHAP/LIME top features (if available)
    top_k = 15
    if SHAP_AVAILABLE and shap_values_dict and feature_names:
        for algo, shap_vals in shap_values_dict.items():
            if len(shap_vals.shape) == 2:  # (n_samples, n_features)
                # Global importance: mean absolute SHAP values
                global_importance = np.abs(shap_vals).mean(axis=0)
            else:
                global_importance = np.abs(shap_vals)

            top_indices = np.argsort(global_importance)[-top_k:][::-1]
            top_features = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in top_indices]
            top_values = global_importance[top_indices]

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(len(top_features)), top_values, color='coral', alpha=0.7)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features)
            ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
            ax.set_title(f'Top {top_k} Features (SHAP) - {algo}', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            path = vis_dir / f"exp_shap_top_features_{algo}.png"
            save_fig(fig, path)
            generated[f'exp_shap_top_features_{algo}'] = path

    if LIME_AVAILABLE and lime_importances_dict and feature_names:
        for algo, lime_imp in lime_importances_dict.items():
            if len(lime_imp) > 0:
                top_indices = np.argsort(np.abs(lime_imp))[-top_k:][::-1]
                top_features = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in top_indices]
                top_values = np.abs(lime_imp[top_indices])

                fig, ax = plt.subplots(figsize=(10, 8))
                ax.barh(range(len(top_features)), top_values, color='lightgreen', alpha=0.7)
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features)
                ax.set_xlabel('Mean |LIME Importance|', fontsize=12)
                ax.set_title(f'Top {top_k} Features (LIME) - {algo}', fontsize=14, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                path = vis_dir / f"exp_lime_top_features_{algo}.png"
                save_fig(fig, path)
                generated[f'exp_lime_top_features_{algo}'] = path

    return generated


# ============================================================================
# TRANSVERSAL 3D VISUALIZATIONS (7)
# ============================================================================

def generate_3d_transversal_visualizations(
    scores_normalized_df: Optional[pd.DataFrame],
    metrics_df: pd.DataFrame,
    vis_dir: Path
) -> Dict[str, Path]:
    """Generate all 3D transversal visualizations"""
    generated = {}

    if scores_normalized_df is None or scores_normalized_df.empty:
        logger.warning("No normalized scores available for 3D visualizations")
        return generated

    algos = scores_normalized_df['algo'] if 'algo' in scores_normalized_df.columns else scores_normalized_df.index
    if len(algos) == 0:
        logger.warning("No algorithms available for 3D visualizations")
        return generated

    perf_scores = scores_normalized_df.get('perf_score', [])
    res_scores = scores_normalized_df.get('resource_score', [])
    exp_scores = scores_normalized_df.get('explain_score', [])

    # 1. dim3_radar.png
    if len(perf_scores) > 0 and len(res_scores) > 0 and len(exp_scores) > 0:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        categories = ['Performance', 'Resources', 'Explainability']
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(algos)))
        for idx, (algo, perf, res, exp) in enumerate(zip(algos, perf_scores, res_scores, exp_scores)):
            values = [perf, res, exp]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=algo, color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title('3D Evaluation Framework - Radar Chart', fontsize=14, fontweight='bold', pad=20)
        path = vis_dir / "dim3_radar.png"
        save_fig(fig, path)
        generated['dim3_radar'] = path

    # 2-4. Scatter plots
    if len(perf_scores) > 0 and len(res_scores) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(res_scores, perf_scores, s=100, alpha=0.6, c='steelblue')
        for algo, res, perf in zip(algos, res_scores, perf_scores):
            ax.annotate(algo, (res, perf), fontsize=9)
        ax.set_xlabel('Resource Score', fontsize=12)
        ax.set_ylabel('Performance Score', fontsize=12)
        ax.set_title('3D Scatter: Performance vs Resources', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        path = vis_dir / "dim3_scatter_perf_res.png"
        save_fig(fig, path)
        generated['dim3_scatter_perf_res'] = path

    if len(perf_scores) > 0 and len(exp_scores) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(exp_scores, perf_scores, s=100, alpha=0.6, c='coral')
        for algo, exp, perf in zip(algos, exp_scores, perf_scores):
            ax.annotate(algo, (exp, perf), fontsize=9)
        ax.set_xlabel('Explainability Score', fontsize=12)
        ax.set_ylabel('Performance Score', fontsize=12)
        ax.set_title('3D Scatter: Performance vs Explainability', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        path = vis_dir / "dim3_scatter_perf_exp.png"
        save_fig(fig, path)
        generated['dim3_scatter_perf_exp'] = path

    if len(res_scores) > 0 and len(exp_scores) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(exp_scores, res_scores, s=100, alpha=0.6, c='lightgreen')
        for algo, exp, res in zip(algos, exp_scores, res_scores):
            ax.annotate(algo, (exp, res), fontsize=9)
        ax.set_xlabel('Explainability Score', fontsize=12)
        ax.set_ylabel('Resource Score', fontsize=12)
        ax.set_title('3D Scatter: Resources vs Explainability', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        path = vis_dir / "dim3_scatter_res_exp.png"
        save_fig(fig, path)
        generated['dim3_scatter_res_exp'] = path

    # 5. dim3_scores_table.png (render table as image)
    if len(perf_scores) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        table_data = []
        for algo, perf, res, exp in zip(algos, perf_scores, res_scores, exp_scores):
            table_data.append([algo, f'{perf:.3f}', f'{res:.3f}', f'{exp:.3f}'])
        table = ax.table(cellText=table_data, colLabels=['Algorithm', 'Performance', 'Resources', 'Explainability'],
                        cellLoc='center', loc='center', bbox=(0, 0, 1, 1))
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax.set_title('3D Evaluation Scores Summary', fontsize=14, fontweight='bold', pad=20)
        path = vis_dir / "dim3_scores_table.png"
        save_fig(fig, path)
        generated['dim3_scores_table'] = path

    return generated


def generate_visualizations_index(generated_files: Dict[str, Path], vis_dir: Path):
    """Generate INDEX.md with all visualizations"""
    index_path = vis_dir / "INDEX.md"

    categories = {
        'Performance': ['perf_'],
        'Resources': ['res_'],
        'Explainability': ['exp_'],
        '3D Transversal': ['dim3_']
    }

    content = "# Phase 3 Visualizations Index\n\n"
    content += f"Total visualizations: {len(generated_files)}\n\n"

    for cat_name, prefixes in categories.items():
        content += f"## {cat_name} Dimension\n\n"
        matching = {k: v for k, v in generated_files.items() if any(k.startswith(p) for p in prefixes)}
        for name, path in sorted(matching.items()):
            filename = path.name
            desc = get_visualization_description(name)
            content += f"- **{filename}**: {desc}\n"
        content += "\n"

    index_path.write_text(content)
    logger.info(f"Generated INDEX.md: {index_path}")


def get_visualization_description(name: str) -> str:
    """Get description for a visualization"""
    descriptions = {
        'perf_f1_bar': 'Bar chart showing mean F1 score with error bars (std)',
        'perf_metrics_grouped_bar': 'Grouped bar chart comparing F1, Precision, Recall, Accuracy',
        'perf_f1_boxplot': 'Boxplot showing F1 score distribution across folds',
        'perf_confusion_matrix': 'Confusion matrix (normalized) for binary/multi-class classification',
        'perf_roc': 'ROC curve with AUC score (binary classification)',
        'perf_pr': 'Precision-Recall curve (binary classification)',
        'perf_metrics_heatmap': 'Heatmap of all performance metrics by algorithm',
        'res_train_time_bar': 'Bar chart showing training time (mean ± std)',
        'res_peak_ram_bar': 'Bar chart showing peak RAM usage (mean ± std)',
        'res_latency_bar': 'Bar chart showing inference latency (mean ± std)',
        'res_tradeoff_f1_vs_time': 'Scatter plot: F1 score vs training time tradeoff',
        'res_tradeoff_f1_vs_ram': 'Scatter plot: F1 score vs peak RAM tradeoff',
        'res_pareto_frontier': 'Pareto frontier highlighting non-dominated algorithms',
        'res_components_heatmap': 'Heatmap of resource components (time, RAM, latency)',
        'exp_score_bar': 'Bar chart showing explainability score by algorithm',
        'exp_components_stacked_bar': 'Stacked bar chart: Native, SHAP, LIME contributions',
        'exp_tradeoff_f1_vs_explain': 'Scatter plot: F1 score vs explainability tradeoff',
        'exp_components_heatmap': 'Heatmap of explainability components by algorithm',
        'exp_shap_top_features': 'Top-k features by mean absolute SHAP values (if SHAP available)',
        'exp_lime_top_features': 'Top-k features by LIME importance (if LIME available)',
        'dim3_radar': 'Radar/spider chart: 3D scores for all algorithms',
        'dim3_scatter_perf_res': 'Scatter plot: Performance vs Resource scores',
        'dim3_scatter_perf_exp': 'Scatter plot: Performance vs Explainability scores',
        'dim3_scatter_res_exp': 'Scatter plot: Resource vs Explainability scores',
        'dim3_scores_table': 'Summary table of all 3D scores rendered as image'
    }

    # Match prefix
    for key, desc in descriptions.items():
        if name.startswith(key) or key in name:
            return desc
    return 'Visualization chart'
