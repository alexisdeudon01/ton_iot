from abc import ABC, abstractmethod
from typing import Any, Dict

class LoggingPort(ABC):
    @abstractmethod
    def info(self, action: str, message: str, **kwargs) -> None:
        pass

    @abstractmethod
    def error(self, action: str, message: str, **kwargs) -> None:
        pass

    @abstractmethod
    def warning(self, action: str, message: str, **kwargs) -> None:
        pass

    @abstractmethod
    def debug(self, action: str, message: str, **kwargs) -> None:
        pass
