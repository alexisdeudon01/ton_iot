import time
from typing import Any, Dict
from src.core.ports.logging import StructuredLogger
from src.core.ports.event_bus import EventBus
from src.core.contracts.events import Event

class StructuredEventLogger(StructuredLogger):
    def __init__(self, event_bus: EventBus, run_id: str):
        self.event_bus = event_bus
        self.run_id = run_id

    def log(self, action: str, message: str, level: str = "INFO", **kwargs) -> None:
        payload = {
            "action": action,
            "message": message,
            "level": level,
            **kwargs
        }
        self.event_bus.publish(Event(
            type="LOG_LINE",
            payload=payload,
            ts=time.time(),
            run_id=self.run_id
        ))

    def info(self, action: str, message: str, **kwargs) -> None:
        self.log(action, message, level="INFO", **kwargs)

    def error(self, action: str, message: str, **kwargs) -> None:
        self.log(action, message, level="ERROR", **kwargs)

    def warning(self, action: str, message: str, **kwargs) -> None:
        self.log(action, message, level="WARNING", **kwargs)
