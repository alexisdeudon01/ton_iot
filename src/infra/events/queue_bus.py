import queue
import threading
from typing import List, Callable
from src.core.ports.event_bus import EventBus
from src.core.contracts.events import Event

class QueueEventBus(EventBus):
    def __init__(self):
        self._queue = queue.Queue()
        self._subscribers: List[Callable[[Event], None]] = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._process_queue, daemon=True)
        self._thread.start()

    def publish(self, event: Event) -> None:
        self._queue.put(event)

    def subscribe(self, callback: Callable[[Event], None]) -> None:
        self._subscribers.append(callback)

    def _process_queue(self):
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.1)
                for subscriber in self._subscribers:
                    try:
                        subscriber(event)
                    except Exception as e:
                        # In a real app, we'd log this to stderr or a fallback logger
                        print(f"Error in event subscriber: {e}")
                self._queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        self._stop_event.set()
        self._thread.join()
