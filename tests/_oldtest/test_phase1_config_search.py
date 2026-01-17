#!/usr/bin/env python3
"""
Tests for Phase 1: Configuration Search
"""
import sys
from pathlib import Path
import pytest

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import generate_108_configs


def test_generate_108_configs():
    """Test that exactly 108 configurations are generated"""
    configs = generate_108_configs()
    assert len(configs) == 108, f"Expected exactly 108 configs (got {len(configs)})"


def test_config_structure():
    """Test that each config has required keys"""
    configs = generate_108_configs()
    
    required_keys = {
        'apply_cleaning',
        'apply_encoding',
        'apply_feature_selection',
        'feature_selection_k',
        'apply_scaling',
        'scaling_method',
        'apply_resampling',
        'resampling_method'
    }
    
    for idx, config in enumerate(configs):
        missing_keys = required_keys - set(config.keys())
        assert len(missing_keys) == 0, \
            f"Config {idx} missing required keys: {missing_keys} (available: {set(config.keys())})"


def test_config_values():
    """Test that config values are valid"""
    configs = generate_108_configs()
    
    valid_scaling = [None, 'RobustScaler', 'StandardScaler']
    valid_resampling = [None, 'SMOTE', 'ADASYN']
    valid_k = [None, 10, 20, 30]
    
    for idx, config in enumerate(configs):
        assert config['apply_cleaning'] in [True, False], \
            f"Config {idx}: apply_cleaning should be bool (got {config['apply_cleaning']})"
        assert config['apply_encoding'] in [True, False], \
            f"Config {idx}: apply_encoding should be bool (got {config['apply_encoding']})"
        assert config['apply_feature_selection'] in [True, False], \
            f"Config {idx}: apply_feature_selection should be bool (got {config['apply_feature_selection']})"
        assert config['apply_scaling'] in [True, False], \
            f"Config {idx}: apply_scaling should be bool (got {config['apply_scaling']})"
        assert config['apply_resampling'] in [True, False], \
            f"Config {idx}: apply_resampling should be bool (got {config['apply_resampling']})"
        
        assert config['scaling_method'] in valid_scaling, \
            f"Config {idx}: scaling_method should be in {valid_scaling} (got {config['scaling_method']})"
        assert config['resampling_method'] in valid_resampling, \
            f"Config {idx}: resampling_method should be in {valid_resampling} (got {config['resampling_method']})"
        assert config['feature_selection_k'] in valid_k, \
            f"Config {idx}: feature_selection_k should be in {valid_k} (got {config['feature_selection_k']})"
        
        # Conditional checks
        if not config['apply_feature_selection']:
            assert config['feature_selection_k'] is None, \
                f"Config {idx}: feature_selection_k should be None when apply_feature_selection=False (got {config['feature_selection_k']})"
        if not config['apply_scaling']:
            assert config['scaling_method'] is None, \
                f"Config {idx}: scaling_method should be None when apply_scaling=False (got {config['scaling_method']})"
        if not config['apply_resampling']:
            assert config['resampling_method'] is None, \
                f"Config {idx}: resampling_method should be None when apply_resampling=False (got {config['resampling_method']})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
