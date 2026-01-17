"""
Test model registry behavior (with/without optional dependencies)
"""
import sys
from pathlib import Path
import pytest

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.models.registry import get_model_registry
from src.config import PipelineConfig


def test_registry_always_available():
    """Test that LR, DT, RF are always available"""
    config = PipelineConfig()
    registry = get_model_registry(config)
    
    assert 'Logistic_Regression' in registry, "Logistic_Regression should always be available in registry"
    assert 'Decision_Tree' in registry, "Decision_Tree should always be available in registry"
    assert 'Random_Forest' in registry, "Random_Forest should always be available in registry"


def test_registry_optional_models():
    """Test that CNN/TabNet are optionally available"""
    config = PipelineConfig()
    registry = get_model_registry(config)
    
    # CNN and TabNet may or may not be available (depends on torch/pytorch-tabnet)
    # Just check that registry is a valid dict
    assert isinstance(registry, dict), f"Registry should be a dict (got {type(registry)})"
    assert len(registry) >= 3, f"Registry should have at least 3 models (got {len(registry)}): LR, DT, RF"
