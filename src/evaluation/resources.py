#!/usr/bin/env python3
"""
Dimension 2: Resource Metrics (training time, peak RAM, inference latency)
"""
import time
import numpy as np
import pandas as pd
import psutil
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Try to import tracemalloc for better RAM tracking
try:
    import tracemalloc
    TRACEMALLOC_AVAILABLE = True
except ImportError:
    TRACEMALLOC_AVAILABLE = False
    logger.warning("tracemalloc not available, using psutil for RAM tracking")


def measure_training_time(model_fit_fn, *args, **kwargs) -> float:
    """
    Measure training time in seconds
    
    Args:
        model_fit_fn: Model fit function (callable)
        *args, **kwargs: Arguments for fit function
        
    Returns:
        Training time in seconds
    """
    start = time.perf_counter()
    model_fit_fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return float(elapsed)


def measure_peak_ram(process_fn, *args, **kwargs) -> float:
    """
    Measure peak RAM usage in MB
    
    Args:
        process_fn: Function to monitor
        *args, **kwargs: Arguments for process_fn
        
    Returns:
        Peak RAM usage in MB
    """
    if TRACEMALLOC_AVAILABLE:
        tracemalloc.start()
        try:
            process_fn(*args, **kwargs)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return float(peak / 1024 / 1024)  # Convert to MB
        except Exception as e:
            tracemalloc.stop()
            logger.warning(f"tracemalloc failed: {e}, falling back to psutil")
    
    # Fallback to psutil RSS peak
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    process_fn(*args, **kwargs)
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    return max(mem_before, mem_after)


def measure_inference_latency(model, X_sample, n_runs: int = 100) -> float:
    """
    Measure inference latency in milliseconds (average over n_runs)
    
    Args:
        model: Trained model with predict() method
        X_sample: Sample input data
        n_runs: Number of inference runs
        
    Returns:
        Average latency in milliseconds
    """
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(X_sample)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
    return float(np.mean(times))


def compute_resource_efficiency(metrics_df, time_col='train_time_sec', 
                                ram_col='peak_ram_mb', latency_col='latency_ms',
                                weights=(0.5, 0.3, 0.2), eps=1e-6) -> pd.DataFrame:
    """
    Compute resource efficiency scores (inverted normalized metrics)
    
    efficiency(x) = 1 - (x - min) / (max - min + eps)
    
    resource_score = w_time * eff(time) + w_ram * eff(ram) + w_latency * eff(latency)
    
    Args:
        metrics_df: DataFrame with resource metrics
        time_col, ram_col, latency_col: Column names
        weights: (w_time, w_ram, w_latency)
        eps: Small epsilon to avoid division by zero
        
    Returns:
        DataFrame with efficiency scores added
    """
    df = metrics_df.copy()
    w_time, w_ram, w_latency = weights
    
    # Compute efficiency for each metric
    for col, w in [(time_col, w_time), (ram_col, w_ram), (latency_col, w_latency)]:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                df[f'eff_{col}'] = 1 - (df[col] - col_min) / (col_max - col_min + eps)
            else:
                df[f'eff_{col}'] = 1.0
    
    # Compute resource score
    df['resource_score'] = (
        w_time * df.get(f'eff_{time_col}', 0) +
        w_ram * df.get(f'eff_{ram_col}', 0) +
        w_latency * df.get(f'eff_{latency_col}', 0)
    )
    
    return df
