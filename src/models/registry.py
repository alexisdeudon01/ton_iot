#!/usr/bin/env python3
"""
Model Registry - Single source of truth for all models
Handles optional dependencies (CNN, TabNet)
"""
import logging
from typing import Dict, Callable, Any, Optional

logger = logging.getLogger(__name__)

# Import sklearn models (always available)
from .sklearn_models import make_lr, make_dt, make_rf

# Try to import CNN (optional)
try:
    from .cnn import CNNTabularClassifier, TORCH_AVAILABLE as CNN_AVAILABLE
except (ImportError, AttributeError):
    CNN_AVAILABLE = False
    CNNTabularClassifier = None

# Try to import TabNet (optional)
try:
    from .tabnet import TabNetClassifierWrapper, TABNET_AVAILABLE
except (ImportError, AttributeError):
    TABNET_AVAILABLE = False
    TabNetClassifierWrapper = None


def get_model_registry(config) -> Dict[str, Callable[[], Any]]:
    """
    Get model registry with all available models
    
    Args:
        config: PipelineConfig instance
        
    Returns:
        Dict mapping model names to builder functions
    """
    registry = {}
    
    # Always available models
    registry['Logistic_Regression'] = lambda: make_lr(random_state=config.random_state, max_iter=1000)
    registry['Decision_Tree'] = lambda: make_dt(random_state=config.random_state)
    registry['Random_Forest'] = lambda: make_rf(random_state=config.random_state, n_estimators=100)
    
    # Optional: CNN
    if CNN_AVAILABLE and CNNTabularClassifier is not None:
        try:
            registry['CNN'] = lambda: CNNTabularClassifier(
                epochs=20,
                batch_size=64,
                random_state=config.random_state
            )
            logger.info("✓ CNN available in model registry")
        except Exception as e:
            logger.warning(f"CNN builder failed: {e}. CNN will be skipped.")
    else:
        logger.warning("CNN skipped (torch not available). Install via: pip install -r requirements-nn.txt")
    
    # Optional: TabNet
    if TABNET_AVAILABLE and TabNetClassifierWrapper is not None:
        try:
            registry['TabNet'] = lambda: TabNetClassifierWrapper(
                max_epochs=50,
                batch_size=1024,
                seed=config.random_state,
                verbose=0
            )
            logger.info("✓ TabNet available in model registry")
        except Exception as e:
            logger.warning(f"TabNet builder failed: {e}. TabNet will be skipped.")
    else:
        logger.warning("TabNet skipped (pytorch-tabnet not available). Install via: pip install -r requirements-nn.txt")
    
    logger.info(f"Model registry initialized with {len(registry)} models: {list(registry.keys())}")
    
    return registry
