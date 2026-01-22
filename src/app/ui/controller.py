import threading
import time
from typing import Callable
from src.app.ui.state import UIState
from src.app.ui.reducer import reduce_event
from src.infra.events.queue_bus import QueueEventBus
from src.app.pipeline.main import run_pipeline
from src.core.contracts.events import Event

class UIController:
    def __init__(self, state_callback: Callable[[UIState], None]):
        self.state = UIState()
        self.state_callback = state_callback
        self.event_bus = QueueEventBus()
        self.event_bus.subscribe(self._handle_event)
        
    def _handle_event(self, event: Event):
        self.state = reduce_event(self.state, event)
        self.state_callback(self.state)

    def start_pipeline(self, config_path: str):
        if self.state.is_running:
            return
        
        thread = threading.Thread(
            target=run_pipeline,
            args=(config_path, self.event_bus),
            daemon=True
        )
        thread.start()

    def request_distribution(self, artifact_id: str, feature: str):
        # In a real app, this would trigger a task in the pipeline
        # For now, we'll simulate a response if the data is available
        pass
