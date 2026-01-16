#!/usr/bin/env python3
"""
Generate CSV exports and Markdown reports for Phase 3
"""
import pandas as pd
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def export_metrics_csvs(metrics_by_fold: pd.DataFrame, metrics_aggregated: pd.DataFrame,
                        scores_normalized: pd.DataFrame, output_dir: Path):
    """
    Export all CSV files for Phase 3
    
    Args:
        metrics_by_fold: Metrics per fold
        metrics_aggregated: Aggregated metrics
        scores_normalized: Normalized scores
        output_dir: Output directory
    """
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_by_fold.to_csv(metrics_dir / "metrics_by_fold.csv", index=False)
    metrics_aggregated.to_csv(metrics_dir / "metrics_aggregated.csv", index=False)
    scores_normalized.to_csv(metrics_dir / "scores_normalized.csv", index=False)
    
    logger.info(f"Exported CSV files to {metrics_dir}")


def generate_algorithm_reports(metrics_df: pd.DataFrame, output_dir: Path):
    """
    Generate Markdown report for each algorithm
    
    Args:
        metrics_df: Metrics DataFrame
        output_dir: Output directory
    """
    reports_dir = output_dir / "algorithm_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    for algo in metrics_df['algo'].unique():
        report_path = reports_dir / f"{algo}.md"
        with open(report_path, 'w') as f:
            f.write(f"# {algo} Evaluation Report\n\n")
            f.write("## Performance Metrics\n\n")
            # Add metrics details
        logger.info(f"Generated report: {report_path}")


def generate_index_md(output_dir: Path):
    """Generate INDEX.md listing all visualizations"""
    index_path = output_dir / "visualizations" / "INDEX.md"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(index_path, 'w') as f:
        f.write("# Phase 3 Visualizations Index\n\n")
        f.write("## Performance (Dimension 1)\n")
        f.write("- `perf_f1_bar.png`: F1 score comparison\n")
        f.write("- `perf_metrics_grouped_bar.png`: All performance metrics grouped\n")
        # Add all 27 visualizations
    logger.info(f"Generated INDEX.md: {index_path}")
