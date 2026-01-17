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
    model_type = type(template).__name__
    logger.debug(f"Creating fresh model from template: {model_type}")
    
    # Try sklearn.clone first (best for sklearn models)
    try:
        from sklearn.base import clone
        fresh = clone(template)
        logger.debug(f"Successfully cloned {model_type} using sklearn.clone()")
        return fresh
    except (TypeError, AttributeError, ValueError) as clone_err:
        logger.debug(f"sklearn.clone() failed for {model_type}: {clone_err}, trying get_params()...")
        
        # Try get_params() approach
        if hasattr(template, 'get_params') and hasattr(template, '__class__'):
            try:
                params = template.get_params()
                model_class = template.__class__
                fresh = model_class(**params)
                logger.debug(f"Successfully recreated {model_type} using get_params()")
                return fresh
            except Exception as params_err:
                logger.debug(f"get_params() approach failed for {model_type}: {params_err}, using deepcopy...")
        
        # Fallback: deepcopy (works but less safe - may copy fitted state)
        try:
            fresh = copy.deepcopy(template)
            logger.warning(f"Used deepcopy for {model_type} - may preserve fitted state, verify model is unfitted")
            return fresh
        except Exception as deepcopy_err:
            logger.error(f"All model cloning methods failed for {model_type}: {deepcopy_err}")
            logger.error("Returning template directly - WARNING: may cause state issues! Model may not be fresh/unfitted.")
            return template
