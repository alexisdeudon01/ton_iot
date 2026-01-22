import time
import psutil
import os
from typing import Dict

class ResourceTracker:
    """
    Mesure les ressources consommées (temps, RAM, CPU) par une étape du pipeline.
    """
    def __init__(self):
        self.start_time = 0
        self.process = psutil.Process(os.getpid())

    def start(self):
        """Démarre le chronomètre."""
        self.start_time = time.time()

    def get_usage(self) -> Dict[str, float]:
        """Retourne l'utilisation actuelle des ressources depuis le dernier start()."""
        mem_info = self.process.memory_info()
        return {
            "duration_s": time.time() - self.start_time,
            "ram_mb": mem_info.rss / (1024 * 1024),
            "cpu_percent": psutil.cpu_percent()
        }
