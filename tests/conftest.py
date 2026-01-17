"""
Pytest fixtures for IRP Pipeline tests
"""
import pytest
import sys
from pathlib import Path
import numpy as np

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import PipelineConfig, TEST_CONFIG
import subprocess


def pytest_sessionstart(session):
    """Check dependencies before starting tests."""
    print("\nChecking dependencies...")
    try:
        from req import check_dependencies
        if not check_dependencies():
            pytest.exit("Missing dependencies. Run 'python3 req.py install' first.")
    except ImportError:
        pytest.exit("req.py not found in project root.")


@pytest.fixture
def config():
    """PipelineConfig fixture for testing"""
    return TEST_CONFIG


@pytest.fixture
def test_config():
    """Backward-compatible fixture name for smoke tests."""
    return TEST_CONFIG


# ============================================================================
# Common Data Fixtures for Model Testing
# ============================================================================

@pytest.fixture
def synthetic_binary_data():
    """
    Synthetic binary classification dataset (X, y) with default seed.
    
    Returns:
        tuple: (X, y) where:
            - X: numpy array (200, 10) float32
            - y: numpy array (200,) int with values in [0, 1]
    """
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples)
    return X, y


@pytest.fixture
def synthetic_multiclass_data():
    """
    Synthetic multiclass classification dataset (X, y) with 3 classes.
    
    Returns:
        tuple: (X, y) where:
            - X: numpy array (300, 10) float32
            - y: numpy array (300,) int with values in [0, 1, 2]
    """
    np.random.seed(42)
    n_samples = 300
    n_features = 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 3, n_samples)
    return X, y


@pytest.fixture
def synthetic_large_binary_data():
    """
    Larger synthetic binary classification dataset (X, y).
    
    Returns:
        tuple: (X, y) where:
            - X: numpy array (400, 10) float32
            - y: numpy array (400,) int with values in [0, 1]
    """
    np.random.seed(42)
    n_samples = 400
    n_features = 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples)
    return X, y


@pytest.fixture
def synthetic_small_data():
    """
    Small synthetic dataset for quick tests (X, y).
    
    Returns:
        tuple: (X, y) where:
            - X: numpy array (100, 10) float32
            - y: numpy array (100,) int with values in [0, 1]
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples)
    return X, y


@pytest.fixture
def synthetic_tabular_dataset():
    """
    Synthetic tabular dataset for PyTorch Dataset testing.
    
    Returns:
        tuple: (X, y) where:
            - X: numpy array (100, 10) float32
            - y: numpy array (100,) int64 with values in [0, 1]
    """
    np.random.seed(42)
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randint(0, 2, 100).astype(np.int64)
    return X, y
