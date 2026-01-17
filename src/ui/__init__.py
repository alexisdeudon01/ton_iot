#!/usr/bin/env python3
"""
UI module - Optional GUI components using Tkinter
All GUI functions are optional and can be safely imported even if Tkinter is unavailable
"""
import logging

logger = logging.getLogger(__name__)

# Try to import GUI components
try:
    from src.ui.features_popup import show_features_popup
    GUI_AVAILABLE = True
except ImportError as e:
    GUI_AVAILABLE = False
    logger.debug(f"GUI not available: {e}")
    
    def show_features_popup(*args, **kwargs):
        """Stub function when GUI is unavailable"""
        logger.debug("GUI popup requested but Tkinter not available - skipping popup")
        pass

__all__ = ['show_features_popup', 'GUI_AVAILABLE']
