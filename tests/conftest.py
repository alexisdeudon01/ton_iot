#!/usr/bin/env python3
"""
Pytest configuration and fixtures
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.config import PipelineConfig, TEST_CONFIG


@pytest.fixture
def test_config():
    """PipelineConfig for testing"""
    return TEST_CONFIG


@pytest.fixture
def sample_config():
    """Standard PipelineConfig for tests"""
    return PipelineConfig(
        test_mode=True,
        sample_ratio=0.001,
        random_state=42,
        output_dir="output/test",
        interactive=False
    )
