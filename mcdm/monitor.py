import psutil
import os
import time
import threading
from typing import Dict

class RealTimeMonitor:
    """
    Moniteur de ressources utilisant psutil pour mesurer l'impact CPU/RAM.
    """
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self.measurements = []

    def _collect(self):
        while not self._stop_event.is_set():
            cpu = self.process.cpu_percent(interval=0.1)
            ram = self.process.memory_info().rss / (1024 * 1024) # MB
            ram_percent = self.process.memory_percent()
            self.measurements.append({
                "cpu": cpu,
                "ram_mb": ram,
                "ram_percent": ram_percent,
                "ts": time.time()
            })
            time.sleep(0.1)

    def start(self):
        self.measurements = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collect)
        self._thread.start()

    def stop(self) -> Dict:
        self._stop_event.set()
        self._thread.join()
        
        if not self.measurements:
            return {"cpu_percent": 0, "ram_percent": 0, "latency_ms": 0}
            
        avg_cpu = sum(m['cpu'] for m in self.measurements) / len(self.measurements)
        avg_ram = sum(m['ram_percent'] for m in self.measurements) / len(self.measurements)
        duration = (self.measurements[-1]['ts'] - self.measurements[0]['ts']) * 1000 # ms
        
        return {
            "cpu_percent": round(avg_cpu, 2),
            "ram_percent": round(avg_ram, 2),
            "latency_ms": round(duration, 2)
        }

def measure_inference(model_func, X_test) -> Dict:
    """
    Mesure les ressources consommées pendant l'inférence d'un modèle.
    """
    monitor = RealTimeMonitor()
    monitor.start()
    
    # Inférence
    _ = model_func(X_test)
    
    stats = monitor.stop()
    return stats
