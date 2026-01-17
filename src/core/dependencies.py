import sys
import os
import logging
import warnings
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

# Export everything for easy access
__all__ = [
    'os', 'sys', 'logging', 'warnings', 'gc', 'pickle', 'time', 'datetime', 'timedelta',
    'Path', 'Dict', 'List', 'Tuple', 'Optional', 'Union', 'Any', 'Set', 'Callable', 'cast',
    'np', 'pd', 'stats', 'tqdm', 'SystemMonitor', 'FeatureAnalyzer', 'IRPFeaturesRequirements'
]
