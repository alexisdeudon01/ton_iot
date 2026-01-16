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


@pytest.fixture
def config():
    """PipelineConfig fixture for testing"""
    return TEST_CONFIG
