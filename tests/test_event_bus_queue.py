import time
import pytest
from src.infra.events.queue_bus import QueueEventBus
from src.core.contracts.events import Event

def test_event_bus_publish_subscribe():
    bus = QueueEventBus()
    received = []
    
    def callback(event):
        received.append(event)
        
    bus.subscribe(callback)
    
    event = Event(
        type="STATUS",
        payload={"message": "test"},
        ts=time.time(),
        run_id="test_run"
    )
    
    bus.publish(event)
    
    # Wait for async processing
    time.sleep(0.2)
    
    assert len(received) == 1
    assert received[0].type == "STATUS"
    assert received[0].payload["message"] == "test"
    
    bus.stop()
