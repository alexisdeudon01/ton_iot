import queue
from typing import List, Callable
from src.core.ports.interfaces import IEventBus
from src.core.events.models import PipelineEvent

class QueueEventBusAdapter(IEventBus):
    def __init__(self):
        self.queue = queue.Queue()
        self.subscribers: List[Callable[[PipelineEvent], None]] = []

    def publish(self, event: PipelineEvent):
        self.queue.put(event)
        for sub in self.subscribers:
            try:
                sub(event)
            except Exception:
                pass # Avoid breaking the publisher if a subscriber fails

    def subscribe(self, callback: Callable[[PipelineEvent], None]):
        self.subscribers.append(callback)

    def get_queue(self) -> queue.Queue:
        return self.queue
