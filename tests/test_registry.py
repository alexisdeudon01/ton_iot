"""
Test model registry behavior (with/without optional dependencies)
"""
import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.models.registry import get_model_registry
from src.config import PipelineConfig


def test_registry_always_available():
    """Test that LR, DT, RF are always available"""
    config = PipelineConfig()
    registry = get_model_registry(config)
    
    assert 'Logistic_Regression' in registry
    assert 'Decision_Tree' in registry
    assert 'Random_Forest' in registry


def test_registry_optional_models():
    """Test that CNN/TabNet are optionally available"""
    config = PipelineConfig()
    registry = get_model_registry(config)
    
    # CNN and TabNet may or may not be available (depends on torch/pytorch-tabnet)
    # Just check that registry is a valid dict
    assert isinstance(registry, dict)
    assert len(registry) >= 3
