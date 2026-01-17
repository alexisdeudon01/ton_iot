"""
Test that core and evaluation modules can be imported without Tkinter
"""
import sys
from pathlib import Path
import pytest

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def test_core_imports():
    """Test core modules import without GUI"""
    from src.core import DatasetLoader, DataHarmonizer, PreprocessingPipeline
    assert DatasetLoader is not None, "DatasetLoader should be importable"
    assert DataHarmonizer is not None, "DataHarmonizer should be importable"
    assert PreprocessingPipeline is not None, "PreprocessingPipeline should be importable"


def test_evaluation_imports():
    """Test evaluation modules import"""
    from src.evaluation.metrics import compute_performance_metrics
    from src.evaluation.resources import measure_training_time
    from src.evaluation.explainability import get_native_interpretability_score
    assert compute_performance_metrics is not None, "compute_performance_metrics should be importable"
    assert measure_training_time is not None, "measure_training_time should be importable"
    assert get_native_interpretability_score is not None, "get_native_interpretability_score should be importable"


def test_models_registry():
    """Test model registry import"""
    from src.models.registry import get_model_registry
    from src.config import PipelineConfig
    config = PipelineConfig()
    registry = get_model_registry(config)
    assert isinstance(registry, dict), f"Registry should be a dict (got {type(registry)})"
    assert len(registry) >= 3, f"Registry should have at least 3 models: LR, DT, RF (got {len(registry)})"
