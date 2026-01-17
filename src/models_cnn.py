#!/usr/bin/env python3
"""
DEPRECATED: Compatibility wrapper for models_cnn.py

This file is deprecated. Please use:
    from src.models.cnn import CNNTabularClassifier, TORCH_AVAILABLE

This wrapper exists for backward compatibility only.
"""
import warnings

warnings.warn(
    "src.models_cnn is deprecated. Use 'from src.models.cnn import CNNTabularClassifier' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from the correct location
from src.models.cnn import CNNTabularClassifier, TORCH_AVAILABLE

# Re-export for backward compatibility
__all__ = ['CNNTabularClassifier', 'TORCH_AVAILABLE']
