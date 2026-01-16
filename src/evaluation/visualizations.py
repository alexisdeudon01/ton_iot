#!/usr/bin/env python3
"""
Generate all Phase 3 visualizations (27 PNG files)
Matplotlib only (no seaborn)
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def save_fig(fig, filepath: Path):
    """Save figure to PNG"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {filepath}")


def generate_all_visualizations(metrics_df: pd.DataFrame, output_dir: Path):
    """
    Generate all 27 Phase 3 visualizations
    
    Args:
        metrics_df: DataFrame with all metrics
        output_dir: Output directory for PNG files
    """
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # DIM 1: Performance
    generate_perf_f1_bar(metrics_df, vis_dir)
    generate_perf_metrics_grouped_bar(metrics_df, vis_dir)
    # ... (stub - will be expanded)
    
    logger.info(f"Generated visualizations in {vis_dir}")


def generate_perf_f1_bar(metrics_df: pd.DataFrame, output_dir: Path):
    """Generate F1 score bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df.plot(x='algo', y='f1_mean', kind='bar', ax=ax, color='steelblue')
    ax.set_title('F1 Score by Algorithm', fontsize=14, fontweight='bold')
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.legend(['F1 Score'])
    plt.xticks(rotation=45, ha='right')
    save_fig(fig, output_dir / "perf_f1_bar.png")


def generate_perf_metrics_grouped_bar(metrics_df: pd.DataFrame, output_dir: Path):
    """Generate grouped bar chart for all performance metrics"""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics_df))
    width = 0.2
    metrics = ['f1_mean', 'precision_mean', 'recall_mean', 'accuracy']
    for i, metric in enumerate(metrics):
        if metric in metrics_df.columns:
            ax.bar(x + i*width, metrics_df[metric], width, label=metric.replace('_mean', '').title())
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics by Algorithm')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics_df['algo'], rotation=45, ha='right')
    ax.legend()
    save_fig(fig, output_dir / "perf_metrics_grouped_bar.png")
