#!/usr/bin/env python3
"""
Smoke tests: vérifier que le pipeline passe sans crash en mode test
"""
import sys
from pathlib import Path
import pytest

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import TEST_CONFIG
from src.app.pipeline_runner import PipelineRunner


def test_pipeline_runner_init():
    """Test PipelineRunner initialization"""
    runner = PipelineRunner(TEST_CONFIG)
    assert runner.config == TEST_CONFIG, "PipelineRunner should use provided config"
    assert runner.results_dir.exists(), f"Results directory should exist at {runner.results_dir}"


@pytest.mark.slow
def test_phase1_smoke(test_config):
    """Smoke test: Phase 1 should run without crashing (may take time)"""
    # Skip if test takes too long - only test structure
    from src.phases.phase1_config_search import Phase1ConfigSearch
    
    phase1 = Phase1ConfigSearch(test_config)
    assert phase1.config == test_config, "Phase1ConfigSearch should use provided config"
    assert len(phase1.configs) == 108, f"Phase 1 should generate exactly 108 configs (got {len(phase1.configs)})"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
