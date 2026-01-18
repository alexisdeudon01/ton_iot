"""
DEPRECATED: This module is kept for backward compatibility only.
Prefer direct imports from standard library, third-party packages, and internal modules.

Example migration:
    OLD: from src.core.dependencies import np, pd, Path
    NEW: import numpy as np; import pandas as pd; from pathlib import Path

    OLD: from src.core.dependencies import SystemMonitor
    NEW: from src.system_monitor import SystemMonitor
"""
import warnings
import sys
import os
import logging
import gc
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable, cast

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Ensure project root is in sys.path to allow absolute imports
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Internal module imports
from src.system_monitor import SystemMonitor
from src.feature_analyzer import FeatureAnalyzer
from src.irp_features_requirements import IRPFeaturesRequirements

# Legacy compatibility imports for verify_irp_compliance.py
try:
    from src.main_pipeline import IRPPipeline
except ImportError:
    IRPPipeline = None

try:
    from src.evaluation_3d import Evaluation3D, ResourceMonitor
except ImportError:
    Evaluation3D = None
    ResourceMonitor = None

try:
    from src.ahp_topsis_framework import AHPTopsisFramework
except ImportError:
    AHPTopsisFramework = None

try:
    from src.realtime_visualizer import RealTimeVisualizer
except ImportError:
    RealTimeVisualizer = None

try:
    from src.results_visualizer import ResultsVisualizer
except ImportError:
    ResultsVisualizer = None

# Core module imports
from src.core import DatasetLoader, DataHarmonizer, PreprocessingPipeline, StratifiedCrossValidator

# Warn on import (once per session)
if not hasattr(sys.modules.get(__name__, None), '_deprecation_warned'):
    warnings.warn(
        "src.core.dependencies is deprecated. Use direct imports instead. "
        "See module docstring for migration guide.",
        DeprecationWarning,
        stacklevel=2
    )
    sys.modules[__name__]._deprecation_warned = True

# Export everything for easy access (backward compatibility)
__all__ = [
    # Standard library
    'os', 'sys', 'logging', 'warnings', 'gc', 'pickle', 'time', 'datetime', 'timedelta',
    # Typing
    'Path', 'Dict', 'List', 'Tuple', 'Optional', 'Union', 'Any', 'Set', 'Callable', 'cast',
    # Third-party
    'np', 'pd', 'stats', 'tqdm',
    # Internal modules
    'SystemMonitor', 'FeatureAnalyzer', 'IRPFeaturesRequirements',
    # Legacy compatibility (may be None if modules don't exist)
    'IRPPipeline', 'Evaluation3D', 'ResourceMonitor', 'AHPTopsisFramework',
    'RealTimeVisualizer', 'ResultsVisualizer',
    # Core modules
    'DatasetLoader', 'DataHarmonizer', 'PreprocessingPipeline', 'StratifiedCrossValidator'
]
