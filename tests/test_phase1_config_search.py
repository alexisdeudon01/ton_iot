#!/usr/bin/env python3
"""
Tests for Phase 1: Configuration Search
"""
import pytest
from src.config import generate_108_configs


def test_generate_108_configs():
    """Test that exactly 108 configurations are generated"""
    configs = generate_108_configs()
    assert len(configs) == 108, f"Expected 108 configs, got {len(configs)}"


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
    
    for config in configs:
        assert all(key in config for key in required_keys), \
            f"Config missing required keys: {set(required_keys) - set(config.keys())}"


def test_config_values():
    """Test that config values are valid"""
    configs = generate_108_configs()
    
    valid_scaling = [None, 'RobustScaler', 'StandardScaler']
    valid_resampling = [None, 'SMOTE', 'ADASYN']
    valid_k = [None, 10, 20, 30]
    
    for config in configs:
        assert config['apply_cleaning'] in [True, False]
        assert config['apply_encoding'] in [True, False]
        assert config['apply_feature_selection'] in [True, False]
        assert config['apply_scaling'] in [True, False]
        assert config['apply_resampling'] in [True, False]
        
        assert config['scaling_method'] in valid_scaling
        assert config['resampling_method'] in valid_resampling
        assert config['feature_selection_k'] in valid_k
        
        # Conditional checks
        if not config['apply_feature_selection']:
            assert config['feature_selection_k'] is None
        if not config['apply_scaling']:
            assert config['scaling_method'] is None
        if not config['apply_resampling']:
            assert config['resampling_method'] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
