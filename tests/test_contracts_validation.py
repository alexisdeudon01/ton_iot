import pytest
from pydantic import ValidationError
from src.core.contracts.config import PipelineConfig, TrainingConfig

def test_pipeline_config_invalid_algorithms():
    # Test that it refuses invalid algorithm list
    with pytest.raises(ValueError, match="Algorithms must be exactly"):
        TrainingConfig(algorithms=["LR", "DT", "RF", "CNN"]) # Missing TabNet

def test_pipeline_config_invalid_order():
    # Test that it refuses invalid order
    with pytest.raises(ValueError, match="Algorithms must be exactly"):
        TrainingConfig(algorithms=["DT", "LR", "RF", "CNN", "TabNet"])

def test_pipeline_config_valid():
    # Test valid config
    cfg = TrainingConfig(algorithms=["LR", "DT", "RF", "CNN", "TabNet"])
    assert cfg.algorithms == ["LR", "DT", "RF", "CNN", "TabNet"]
