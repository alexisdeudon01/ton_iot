#!/usr/bin/env python3
"""
Test that early_fusion adds dataset_source column encoded as int (0=CIC, 1=TON)
"""
import pandas as pd
import pytest

from src.core.data_harmonization import DataHarmonizer


def test_dataset_source_encoded():
    """Test that early_fusion encodes dataset_source as int (0=CIC, 1=TON)."""
    harmonizer = DataHarmonizer()
    
    # Create sample dataframes
    df_cic = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'label': [0, 1, 0]
    })
    
    df_ton = pd.DataFrame({
        'feature1': [7, 8, 9],
        'feature2': [10, 11, 12],
        'label': [1, 0, 1]
    })
    
    # Harmonize and fuse
    df_cic_harm, df_ton_harm = harmonizer.harmonize_features(df_cic, df_ton)
    df_fused, _ = harmonizer.early_fusion(df_cic_harm, df_ton_harm)
    
    # Check that dataset_source column exists
    assert 'dataset_source' in df_fused.columns, "dataset_source column not found"
    
    # Check that dataset_source is encoded as int
    assert df_fused['dataset_source'].dtype in [int, 'int64', 'int32'], \
        f"dataset_source should be int, got {df_fused['dataset_source'].dtype}"
    
    # Check that CIC rows have dataset_source=0
    cic_mask = df_fused.index < len(df_cic)
    assert (df_fused.loc[cic_mask, 'dataset_source'] == 0).all(), \
        "CIC rows should have dataset_source=0"
    
    # Check that TON rows have dataset_source=1
    ton_mask = df_fused.index >= len(df_cic)
    assert (df_fused.loc[ton_mask, 'dataset_source'] == 1).all(), \
        "TON rows should have dataset_source=1"
    
    # Check that values are 0 or 1 only
    unique_values = df_fused['dataset_source'].unique()
    assert set(unique_values) == {0, 1}, \
        f"dataset_source should only contain 0 and 1, got {unique_values}"
