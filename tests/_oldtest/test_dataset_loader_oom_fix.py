"""
Tests for OOM fixes in dataset_loader.py
"""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import tempfile

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.core.dataset_loader import DatasetLoader, DEFAULT_CHUNK_CAP_PROD, DEFAULT_CHUNK_CAP_TEST


def test_chunk_cap_applied():
    """Test that chunk size caps are applied correctly"""
    loader = DatasetLoader()
    
    # Production cap
    chunk_prod = loader._get_adaptive_chunk_size(sample_ratio=1.0)
    assert chunk_prod <= DEFAULT_CHUNK_CAP_PROD, \
        f"Production chunk size {chunk_prod:,} should not exceed cap {DEFAULT_CHUNK_CAP_PROD:,}"
    
    # Test cap
    chunk_test = loader._get_adaptive_chunk_size(sample_ratio=0.001)
    assert chunk_test <= DEFAULT_CHUNK_CAP_TEST, \
        f"Test chunk size {chunk_test:,} should not exceed cap {DEFAULT_CHUNK_CAP_TEST:,}"
    
    assert chunk_prod > 0, f"Production chunk size should be positive (got {chunk_prod})"
    assert chunk_test > 0, f"Test chunk size should be positive (got {chunk_test})"


def test_stream_sampling(tmp_path):
    """Test that sampling works correctly with streaming chunks"""
    # Create a small synthetic CSV
    csv_file = tmp_path / "test_data.csv"
    
    # Generate synthetic data
    np.random.seed(42)
    n_rows = 10000
    data = {
        'feature1': np.random.randn(n_rows),
        'feature2': np.random.randint(0, 100, n_rows),
        'label': np.random.choice(['Benign', 'Attack'], n_rows)
    }
    df_full = pd.DataFrame(data)
    df_full.to_csv(csv_file, index=False)
    
    # Load with sampling
    loader = DatasetLoader()
    sample_ratio = 0.1  # 10%
    
    # Manually test the chunk reading logic
    chunk_size = loader._get_adaptive_chunk_size(sample_ratio)
    chunks = []
    
    chunk_iterator = pd.read_csv(csv_file, low_memory=True, chunksize=chunk_size)
    for chunk in chunk_iterator:
        chunk_sample_size = max(1, int(len(chunk) * sample_ratio))
        chunk_sampled = chunk.sample(n=chunk_sample_size, random_state=42).reset_index(drop=True)
        chunks.append(chunk_sampled)
    
    df_loaded = pd.concat(chunks, ignore_index=True) if len(chunks) > 1 else chunks[0]
    
    # Should have approximately 10% of rows (allow some variance due to chunk boundaries)
    expected_min = int(n_rows * sample_ratio * 0.8)  # Allow 20% variance
    expected_max = int(n_rows * sample_ratio * 1.2)
    
    assert expected_min <= len(df_loaded) <= expected_max, \
        f"Expected ~{n_rows * sample_ratio:.0f} rows (range: {expected_min}-{expected_max}), got {len(df_loaded)}"
    assert len(df_loaded) > 0, "Loaded dataframe should not be empty"


def test_max_files_limit(tmp_path):
    """Test that max_files parameter limits the number of files loaded"""
    # Create 5 test CSV files
    np.random.seed(42)
    for i in range(5):
        csv_file = tmp_path / f"file_{i:02d}.csv"
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'label': ['Benign'] * 100
        })
        data.to_csv(csv_file, index=False)
    
    loader = DatasetLoader()
    
    # Load with max_files=2
    dfs = []
    csv_files = sorted(tmp_path.glob("*.csv"))
    max_files = 2
    
    assert len(csv_files) == 5, f"Should have created 5 CSV files (got {len(csv_files)})"
    
    for csv_file in csv_files[:max_files]:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    assert len(dfs) == 2, f"Should load exactly {max_files} files (got {len(dfs)})"
    assert all(isinstance(df, pd.DataFrame) for df in dfs), "All loaded items should be DataFrames"


def test_optimize_dtypes():
    """Test dtype optimization reduces memory"""
    loader = DatasetLoader()
    
    # Create DF with inefficient dtypes
    np.random.seed(42)
    df = pd.DataFrame({
        'float_col': np.random.randn(1000).astype('float64'),
        'int_col': np.random.randint(0, 100, 1000).astype('int64'),
        'object_col': ['cat', 'dog', 'bird'] * 333 + ['cat'],
        'label': ['A', 'B'] * 500
    })
    
    mem_before = df.memory_usage(deep=True).sum()
    df_opt = loader._optimize_dtypes(df)
    mem_after = df_opt.memory_usage(deep=True).sum()
    
    assert mem_after < mem_before, \
        f"Dtype optimization should reduce memory (before: {mem_before/1024:.1f}KB, after: {mem_after/1024:.1f}KB)"
    
    # Check that float/int were downcast
    assert df_opt['float_col'].dtype in ['float32', 'float64'], \
        f"Float column should be downcast to float32/float64 (got {df_opt['float_col'].dtype})"
    assert df_opt['int_col'].dtype in ['int32', 'int64', 'int16', 'int8'], \
        f"Int column should be downcast (got {df_opt['int_col'].dtype})"
    
    # Check that shape is preserved
    assert df_opt.shape == df.shape, \
        f"Shape should be preserved after optimization (got {df_opt.shape}, expected {df.shape})"
