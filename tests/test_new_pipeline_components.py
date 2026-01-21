import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.new_pipeline.data_loader import RealDataLoader
from src.system_monitor import SystemMonitor
from src.core.feature_categorization import categorize_features, get_category_scores
from src.new_pipeline.trainer import PipelineTrainer
from src.new_pipeline.validator import PipelineValidator
from src.new_pipeline.xai_manager import XAIManager
from src.new_pipeline.tester import PipelineTester

@pytest.fixture
def monitor():
    return SystemMonitor(max_memory_percent=90.0) # High limit for tests

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Flow Duration': [100, 200, 300],
        'Total Fwd Packets': [10, 20, 30],
        'is_ddos': [0, 1, 0],
        'type': [0, 1, 0],
        'Label': ['BENIGN', 'ddos', 'BENIGN']
    })

def test_feature_categorization():
    cols = ['Flow Duration', 'Source IP', 'SYN Flag Count', 'Unknown_Col']
    categorized = categorize_features(cols)
    assert 'Flow_Basic_Stats' in categorized
    assert 'Flow Duration' in categorized['Flow_Basic_Stats']
    assert 'Flow_Identifiers' in categorized
    assert 'Source IP' in categorized['Flow_Identifiers']
    assert 'Flag_Counts' in categorized
    assert 'SYN Flag Count' in categorized['Flag_Counts']
    assert 'Other' in categorized
    assert 'Unknown_Col' in categorized['Other']

def test_category_scores():
    cols = ['Flow Duration', 'Source IP']
    categorized = categorize_features(cols)
    scores = get_category_scores(categorized)
    assert 'performance' in scores
    assert 'explainability' in scores
    assert 'resources' in scores
    assert scores['explainability'] > 0

def test_trainer_single(sample_df):
    trainer = PipelineTrainer(random_state=42)
    X = sample_df.drop(['is_ddos', 'type', 'Label'], axis=1)
    y = sample_df['is_ddos']
    trainer.train_single('DT', X, y)
    assert 'DT' in trainer.models
    assert trainer.models['DT'] is not None
    assert 'DT' in trainer.training_times

def test_validator_single(sample_df, tmp_path):
    trainer = PipelineTrainer(random_state=42)
    X = sample_df.drop(['is_ddos', 'type', 'Label'], axis=1)
    y = sample_df['is_ddos']
    trainer.train_single('DT', X, y)

    validator = PipelineValidator(trainer.models, random_state=42)
    validator.validate_tuning(X, y, tmp_path, algo_name='DT')
    assert 'DT' in validator.best_params

def test_xai_manager_single(sample_df, tmp_path):
    trainer = PipelineTrainer(random_state=42)
    X = sample_df.drop(['is_ddos', 'type', 'Label'], axis=1)
    y = sample_df['is_ddos']
    trainer.train_single('RF', X, y)

    xai = XAIManager(rr_dir=tmp_path)
    xai.validate_xai(trainer.models, X, y, algo_name='RF')
    assert 'RF' in xai.results

def test_tester_single(sample_df, tmp_path):
    trainer = PipelineTrainer(random_state=42)
    X = sample_df.drop(['is_ddos', 'type', 'Label'], axis=1)
    y = sample_df['is_ddos']
    trainer.train_single('DT', X, y)

    tester = PipelineTester(trainer.models, rr_dir=tmp_path)
    tester.evaluate_all(X, y, algo_name='DT')
    assert 'DT' in tester.test_results
