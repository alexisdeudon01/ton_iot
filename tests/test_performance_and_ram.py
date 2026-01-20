import pytest
import pandas as pd
import numpy as np
import psutil
import os
from src.core.dataset_loader import DatasetLoader
from src.models.sklearn_models import make_lr, make_rf

def test_ram_optimization_during_loading():
    """Verify that dtype optimization actually reduces RAM usage."""
    loader = DatasetLoader()

    # Create a large-ish dataframe with inefficient types
    n_rows = 100000
    df = pd.DataFrame({
        'float_col': np.random.randn(n_rows).astype('float64'),
        'int_col': np.random.randint(0, 1000, n_rows).astype('int64'),
        'cat_col': np.random.choice(['A', 'B', 'C', 'D'], n_rows)
    })

    initial_mem = df.memory_usage(deep=True).sum()

    # Apply optimization
    df_opt = loader._optimize_dtypes(df)

    optimized_mem = df_opt.memory_usage(deep=True).sum()

    print(f"Initial RAM: {initial_mem / 1024**2:.2f} MB")
    print(f"Optimized RAM: {optimized_mem / 1024**2:.2f} MB")

    assert optimized_mem < initial_mem, "RAM usage should be reduced after optimization"
    assert df_opt['float_col'].dtype == 'float32', "Float64 should be downcast to float32"
    assert df_opt['int_col'].dtype in ['int16', 'int32'], "Int64 should be downcast"

def test_multi_threading_config():
    """Verify that sklearn models are configured to use multi-threading (n_jobs=-1)."""
    # Test Random Forest
    rf = make_rf()
    assert rf.n_jobs == -1, "Random Forest should use all available cores (n_jobs=-1)"

    # Test Logistic Regression
    lr = make_lr()
    assert lr.n_jobs == -1, "Logistic Regression should use all available cores (n_jobs=-1)"

def test_system_monitor_ram_check():
    """Verify that SystemMonitor correctly reports RAM usage."""
    from src.system_monitor import SystemMonitor
    monitor = SystemMonitor()

    mem_info = monitor.get_memory_info()
    assert 'used_percent' in mem_info
    assert 'process_mem_mb' in mem_info
    assert mem_info['process_mem_mb'] > 0

    is_safe, msg = monitor.check_memory_safe()
    assert isinstance(is_safe, bool)
    assert isinstance(msg, str)
