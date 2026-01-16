"""
Test that core and evaluation modules can be imported without Tkinter
"""
import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def test_core_imports():
    """Test core modules import without GUI"""
    from src.core import DatasetLoader, DataHarmonizer, PreprocessingPipeline
    assert DatasetLoader is not None
    assert DataHarmonizer is not None
    assert PreprocessingPipeline is not None


def test_evaluation_imports():
    """Test evaluation modules import"""
    from src.evaluation.metrics import compute_performance_metrics
    from src.evaluation.resources import measure_training_time
    from src.evaluation.explainability import get_native_interpretability_score
    assert compute_performance_metrics is not None
    assert measure_training_time is not None
    assert get_native_interpretability_score is not None


def test_models_registry():
    """Test model registry import"""
    from src.models.registry import get_model_registry
    from src.config import PipelineConfig
    config = PipelineConfig()
    registry = get_model_registry(config)
    assert isinstance(registry, dict)
    assert len(registry) >= 3  # At least LR, DT, RF
