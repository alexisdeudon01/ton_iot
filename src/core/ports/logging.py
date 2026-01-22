from abc import ABC, abstractmethod
from typing import Any, Dict

class StructuredLogger(ABC):
    @abstractmethod
    def log(self, action: str, message: str, level: str = "INFO", **kwargs) -> None:
        pass

    @abstractmethod
    def info(self, action: str, message: str, **kwargs) -> None:
        pass

    @abstractmethod
    def error(self, action: str, message: str, **kwargs) -> None:
        pass

    @abstractmethod
    def warning(self, action: str, message: str, **kwargs) -> None:
        pass
