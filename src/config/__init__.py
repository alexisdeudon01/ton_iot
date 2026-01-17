"""
Configuration module for IRP Pipeline.
This package provides access to the configuration defined in src/config.py.
"""

import importlib.util
import sys
from pathlib import Path

# Import PipelineConfig from parent config.py module (src/config.py, not this package)
# This is necessary because of the naming conflict between src/config.py and src/config/

_current_dir = Path(__file__).parent
_config_path = _current_dir.parent / "config.py"

if not _config_path.exists():
    raise ImportError(f"Configuration file not found at {_config_path}")

# Load the module dynamically
spec = importlib.util.spec_from_file_location("src_config_module", _config_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load spec for configuration at {_config_path}")

config_module = importlib.util.module_from_spec(spec)
# Add to sys.modules to allow the module to be recognized by other parts of the system
sys.modules["src_config_module"] = config_module
spec.loader.exec_module(config_module)

# Export the required members
PipelineConfig = config_module.PipelineConfig
generate_108_configs = config_module.generate_108_configs
TEST_CONFIG = config_module.TEST_CONFIG

__all__ = ["PipelineConfig", "generate_108_configs", "TEST_CONFIG"]
