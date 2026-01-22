import psutil
import os
import time
from typing import Dict
from src.core.ports.event_bus import EventBusPort
from src.core.contracts.events import Event

class ResourceMonitor:
    def __init__(self, event_bus: EventBusPort, run_id: str):
        self.event_bus = event_bus
        self.run_id = run_id
        self.process = psutil.Process(os.getpid())

    def snapshot(self, task_name: str = None) -> Dict[str, float]:
        mem = self.process.memory_info()
        sys_mem = psutil.virtual_memory()
        
        snapshot = {
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": sys_mem.percent,
            "ram_used_mb": mem.rss / (1024 * 1024),
            "ram_avail_mb": sys_mem.available / (1024 * 1024),
            "rss_mb": mem.rss / (1024 * 1024),
            "peak_rss_mb": 0.0 # Simplified
        }
        
        self.event_bus.publish(Event(
            type="RESOURCE_SNAPSHOT",
            payload=snapshot,
            ts=time.time(),
            run_id=self.run_id,
            task_name=task_name
        ))
        return snapshot
