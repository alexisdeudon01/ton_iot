"""
Configuration module for IRP Pipeline
"""
# Import PipelineConfig from parent config.py module (src/config.py, not this package)
import sys
from pathlib import Path

# Import from the module config.py in parent directory
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# Import from the module src.config (config.py file), not this package
import importlib.util
_config_path = _parent_dir / 'config.py'
spec = importlib.util.spec_from_file_location("config_module", _config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)

PipelineConfig = config_module.PipelineConfig
generate_108_configs = config_module.generate_108_configs
TEST_CONFIG = config_module.TEST_CONFIG

__all__ = ['PipelineConfig', 'generate_108_configs', 'TEST_CONFIG']
