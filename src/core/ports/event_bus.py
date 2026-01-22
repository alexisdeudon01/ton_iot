from abc import ABC, abstractmethod
from src.core.contracts.events import Event

class EventBus(ABC):
    @abstractmethod
    def publish(self, event: Event) -> None:
        pass

    @abstractmethod
    def subscribe(self, callback) -> None:
        pass
