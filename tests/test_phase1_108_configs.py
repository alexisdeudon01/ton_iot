"""
Test Phase 1 generates exactly 108 configurations
"""
import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import generate_108_configs


def test_108_configs_generated():
    """Test that generate_108_configs returns exactly 108 configs"""
    configs = generate_108_configs()
    assert len(configs) == 108, f"Expected 108 configs, got {len(configs)}"


def test_configs_have_required_keys():
    """Test that each config has required preprocessing keys"""
    configs = generate_108_configs()
    required_keys = ['apply_encoding', 'apply_feature_selection', 'apply_scaling', 
                     'apply_resampling', 'apply_cleaning']
    
    for config in configs:
        for key in required_keys:
            assert key in config, f"Config missing key: {key}"
