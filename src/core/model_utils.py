#!/usr/bin/env python3
"""
Model utilities - helper functions for model cloning/creation
"""
import logging
from typing import Any
import copy

logger = logging.getLogger(__name__)


def fresh_model(template: Any) -> Any:
    """
    Create a fresh, unfitted model instance from a template
    
    Handles different model types:
    - sklearn models: use clone()
    - Custom models with get_params(): recreate from params
    - Others: deepcopy
    
    Args:
        template: Model instance (fitted or unfitted)
        
    Returns:
        Fresh, unfitted model instance
    """
    # Try sklearn.clone first (best for sklearn models)
    try:
        from sklearn.base import clone
        return clone(template)
    except (TypeError, AttributeError, ValueError) as clone_err:
        logger.debug(f"clone() failed: {clone_err}, trying get_params()...")
        
        # Try get_params() approach
        if hasattr(template, 'get_params') and hasattr(template, '__class__'):
            try:
                params = template.get_params()
                model_class = template.__class__
                return model_class(**params)
            except Exception as params_err:
                logger.debug(f"get_params() approach failed: {params_err}, using deepcopy...")
        
        # Fallback: deepcopy (works but less safe - may copy fitted state)
        try:
            return copy.deepcopy(template)
        except Exception as deepcopy_err:
            logger.warning(f"All model cloning methods failed for {type(template).__name__}: {deepcopy_err}")
            logger.warning("Returning template directly - WARNING: may cause state issues!")
            return template
