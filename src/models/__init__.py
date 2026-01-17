"""
Models package - All ML/DL models together
"""

import sys
from pathlib import Path

from .registry import get_model_registry

__all__ = ["get_model_registry"]
