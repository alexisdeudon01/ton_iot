"""Common visualization helpers"""
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


def save_fig(fig, filepath: Path, dpi: int = 150, bbox_inches: str = 'tight') -> None:
    """
    Save matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure
        filepath: Path to save figure
        dpi: Resolution (default: 150)
        bbox_inches: Bounding box (default: 'tight')
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available, cannot save figure")
        return
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)
    logger.debug(f"Saved figure to {filepath}")


def get_standard_colors() -> Dict[str, str]:
    """
    Get standard color scheme for visualizations.
    
    Returns:
        Dictionary of color names to hex codes
    """
    return {
        'blue': '#1f77b4',
        'orange': '#ff7f0e',
        'green': '#2ca02c',
        'red': '#d62728',
        'purple': '#9467bd',
        'brown': '#8c564b',
        'pink': '#e377c2',
        'gray': '#7f7f7f',
        'olive': '#bcbd22',
        'cyan': '#17becf'
    }


def get_color_scheme(scheme_name: str = 'default') -> list:
    """
    Get color scheme list for plots.
    
    Args:
        scheme_name: Name of color scheme ('default', 'deep_learning', 'tree', etc.)
        
    Returns:
        List of hex color codes
    """
    colors = get_standard_colors()
    
    if scheme_name == 'deep_learning':
        return [colors['blue'], colors['orange'], colors['green'], colors['red']]
    elif scheme_name == 'tree':
        return [colors['green'], colors['olive'], colors['cyan'], colors['blue']]
    else:  # default
        return [colors['blue'], colors['orange'], colors['green'], colors['red']]
