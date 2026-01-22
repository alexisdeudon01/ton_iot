import time
import logging
from typing import Any, Optional
from src.core.ports.logging import LoggingPort
from src.core.ports.event_bus import EventBusPort
from src.core.contracts.events import Event

class StructuredEventLogger(LoggingPort):
    def __init__(self, event_bus: EventBusPort, run_id: str):
        self.event_bus = event_bus
        self.run_id = run_id
        self._logger = logging.getLogger("ddos_pipeline")

    def _publish(self, level: str, action: str, message: str, **kwargs):
        payload = {
            "level": level,
            "action": action,
            "message": message,
            **kwargs
        }
        self.event_bus.publish(Event(
            type="LOG_LINE",
            payload=payload,
            ts=time.time(),
            run_id=self.run_id
        ))
        
        # Emit STATUS event for warnings and errors
        if level in ["WARNING", "ERROR"]:
            self.event_bus.publish(Event(
                type="STATUS",
                payload={
                    "message": f"[{action}] {message}",
                    "severity": "warning" if level == "WARNING" else "error",
                    "details": kwargs
                },
                ts=time.time(),
                run_id=self.run_id
            ))

    def info(self, action: str, message: str, **kwargs):
        self._logger.info(f"[{action}] {message}")
        self._publish("INFO", action, message, **kwargs)

    def warning(self, action: str, message: str, **kwargs):
        self._logger.warning(f"[{action}] {message}")
        self._publish("WARNING", action, message, **kwargs)

    def error(self, action: str, message: str, **kwargs):
        self._logger.error(f"[{action}] {message}")
        self._publish("ERROR", action, message, **kwargs)

    def debug(self, action: str, message: str, **kwargs):
        self._logger.debug(f"[{action}] {message}")
        self._publish("DEBUG", action, message, **kwargs)
