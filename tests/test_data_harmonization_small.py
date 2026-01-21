import pytest
import pandas as pd
import numpy as np
from src.core.data_harmonization import DataHarmonizer

def test_analyze_feature_similarity_basic():
    """Test statistical similarity analysis on small synthetic data."""
    harmonizer = DataHarmonizer()

    df1 = pd.DataFrame({'feat': [1.0, 2.0, 3.0, 4.0, 5.0]})
    df2 = pd.DataFrame({'feat': [1.1, 1.9, 3.2, 3.8, 5.1]})

    result = harmonizer.analyze_feature_similarity(df1, df2, 'feat', 'feat')

    assert result['compatible'] is True
    assert 'ks_pvalue' in result
    assert result['mean_diff'] < 0.5

def test_early_fusion_basic():
    """Test early fusion logic with minimal data."""
    harmonizer = DataHarmonizer()

    df_cic = pd.DataFrame({'f1': [1, 2], 'label': [0, 1]})
    df_ton = pd.DataFrame({'f1': [3, 4], 'label': [0, 1]})

    fused, validation = harmonizer.early_fusion(df_cic, df_ton, validate=True)

    assert len(fused) == 4
    assert 'dataset_source' in fused.columns
    assert list(fused['dataset_source'].unique()) == [0, 1]
    assert 'f1' in validation

def test_harmonize_features_labels():
    """Test label harmonization (binary conversion)."""
    harmonizer = DataHarmonizer()

    df_cic = pd.DataFrame({'F1': [1, 2], 'Label': ['Benign', 'DDoS']})
    df_ton = pd.DataFrame({'f1': [3, 4], 'type': ['normal', 'ddos'], 'label': [0, 1]})

    # Mock feature mapping
    mapping = [
        {'unified_name': 'f1', 'cic_name': 'F1', 'ton_name': 'f1', 'type': 'exact'}
    ]

    h_cic, h_ton = harmonizer.harmonize_features(
        df_cic, df_ton,
        label_col_cic='Label',
        label_col_ton='label',
        precomputed_feature_mapping=mapping
    )

    assert h_cic['label'].tolist() == [0, 1]
    assert h_ton['label'].tolist() == [0, 1]
    assert 'f1' in h_cic.columns
    assert 'f1' in h_ton.columns
