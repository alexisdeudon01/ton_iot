#!/usr/bin/env python3
"""
Test Phase 2 outputs: parquet/csv.gz, feature_names.json, summary.md, dataset_source encoded
"""
import json
import pytest
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import PipelineConfig
from src.phases.phase2_apply_best_config import Phase2ApplyBestConfig


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    return str(tmp_path / "output")


@pytest.fixture
def test_config(temp_output_dir):
    """Create test configuration."""
    return PipelineConfig(
        test_mode=True,
        sample_ratio=0.01,
        random_state=42,
        output_dir=temp_output_dir
    )


@pytest.fixture
def mock_best_config():
    """Create mock best config."""
    return {
        'apply_encoding': True,
        'apply_feature_selection': True,
        'apply_scaling': True,
        'apply_resampling': True,
        'feature_selection_k': 20
    }


def test_phase2_outputs_exist(test_config, mock_best_config, tmp_path, monkeypatch):
    """Test that Phase 2 generates all required outputs."""
    # Mock dataset loading
    def mock_load_ton_iot(*args, **kwargs):
        return pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'label': np.random.randint(0, 2, 100)
        })
    
    def mock_load_cic(*args, **kwargs):
        return pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'label': np.random.randint(0, 2, 50)
        })
    
    # Create Phase 2 instance
    phase2 = Phase2ApplyBestConfig(test_config, mock_best_config)
    
    # Mock the loader methods
    monkeypatch.setattr(phase2.loader, 'load_ton_iot', mock_load_ton_iot)
    monkeypatch.setattr(phase2.loader, 'load_cic_ddos2019', mock_load_cic)
    
    # Run Phase 2
    result = phase2.run()
    
    # Check outputs exist
    results_dir = Path(test_config.output_dir) / 'phase2_apply_best_config'
    
    # Check preprocessed data file (parquet or csv.gz)
    parquet_file = results_dir / 'best_preprocessed.parquet'
    csv_file = results_dir / 'best_preprocessed.csv.gz'
    assert parquet_file.exists() or csv_file.exists(), \
        "Preprocessed data file (parquet or csv.gz) not found"
    
    # Check feature_names.json
    feature_names_file = results_dir / 'feature_names.json'
    assert feature_names_file.exists(), "feature_names.json not found"
    
    # Check phase2_summary.md
    summary_file = results_dir / 'phase2_summary.md'
    assert summary_file.exists(), "phase2_summary.md not found"
    
    # Verify feature_names.json content
    with open(feature_names_file, 'r') as f:
        feature_data = json.load(f)
    assert 'feature_names' in feature_data, "feature_names.json missing 'feature_names' key"
    assert isinstance(feature_data['feature_names'], list), \
        "feature_names should be a list"
    
    # Verify dataset_source is encoded in preprocessed data
    if parquet_file.exists():
        df = pd.read_parquet(parquet_file)
    else:
        df = pd.read_csv(csv_file, compression='gzip')
    
    if 'dataset_source' in df.columns:
        assert df['dataset_source'].dtype in [int, 'int64', 'int32'], \
            f"dataset_source should be int, got {df['dataset_source'].dtype}"
        unique_values = df['dataset_source'].unique()
        assert set(unique_values).issubset({0, 1}), \
            f"dataset_source should only contain 0 and 1, got {unique_values}"
