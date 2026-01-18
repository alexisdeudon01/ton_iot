"""Utilities for optional imports"""
from typing import Any, Tuple, Optional


def optional_import(module_name: str, default: Any = None) -> Tuple[Any, bool]:
    """
    Attempt to import an optional module.
    
    Args:
        module_name: Name of the module to import
        default: Default value to return if import fails
        
    Returns:
        Tuple of (module or default, success boolean)
    """
    try:
        module = __import__(module_name)
        # Handle submodules (e.g., 'sklearn.metrics' -> get metrics)
        if '.' in module_name:
            parts = module_name.split('.')
            for part in parts[1:]:
                module = getattr(module, part)
        return module, True
    except ImportError:
        return default, False


def check_optional_import(module_name: str) -> bool:
    """
    Check if an optional module is available.
    
    Args:
        module_name: Name of the module to check
        
    Returns:
        True if module is available, False otherwise
    """
    _, available = optional_import(module_name)
    return available
