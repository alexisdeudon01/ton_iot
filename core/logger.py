import logging
import json
from datetime import datetime
from typing import Any, Dict

class StructuredLogger:
    """
    Logger produisant des sorties structurées pour la traçabilité et l'affichage GUI.
    """
    def __init__(self, name: str = "DDoSPipeline"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _log(self, level: str, action: str, message: str, **kwargs):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "action": action,
            "message": message,
            "metadata": kwargs
        }
        self.logger.info(json.dumps(log_entry))

    def info(self, action: str, message: str, **kwargs):
        self._log("INFO", action, message, **kwargs)

    def warning(self, action: str, message: str, **kwargs):
        self._log("WARNING", action, message, **kwargs)

    def error(self, action: str, message: str, **kwargs):
        self._log("ERROR", action, message, **kwargs)

pipeline_logger = StructuredLogger()
