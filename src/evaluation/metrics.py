#!/usr/bin/env python3
"""
Dimension 1: Performance Metrics (F1, Precision, Recall, Accuracy)
"""
import numpy as np
from src.datastructure.toniot_dataframe import ToniotDataFrame
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from typing import Dict, List, Tuple, Literal, cast
import logging

logger = logging.getLogger(__name__)


def compute_performance_metrics(y_true, y_pred, average: str = 'weighted') -> Dict[str, float]:
    """
    Compute performance metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'binary' for 2 classes, 'weighted' otherwise

    Returns:
        Dict with f1, precision, recall, accuracy
    """
    avg = cast(Literal['micro', 'macro', 'samples', 'weighted', 'binary'], average)
    return {
        'f1': float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred))
    }


def aggregate_metrics_per_algorithm(metrics_by_fold: ToniotDataFrame) -> ToniotDataFrame:
    """
    Aggregate metrics across folds per algorithm

    Args:
        metrics_by_fold: DataFrame with columns [algo, fold, f1, precision, recall, accuracy, ...]

    Returns:
        DataFrame with mean/std per algorithm
    """
    agg_metrics = []
    for algo in metrics_by_fold['algo'].unique():
        df_algo = metrics_by_fold[metrics_by_fold['algo'] == algo]
        row = {'algo': algo}
        for col in ['f1', 'precision', 'recall', 'accuracy']:
            if col in df_algo.columns:
                row[f'{col}_mean'] = float(df_algo[col].mean())
                row[f'{col}_std'] = float(df_algo[col].std())
        agg_metrics.append(row)
    return ToniotDataFrame(agg_metrics)
