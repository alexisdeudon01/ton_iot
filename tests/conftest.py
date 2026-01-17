"""
Pytest fixtures for IRP Pipeline tests
"""
import pytest
import sys
from pathlib import Path

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
