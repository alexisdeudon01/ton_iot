import psutil
import os
import gc
from typing import Optional
from config.schema import MemoryConfig

class MemoryManager:
    """
    Gère l'utilisation de la RAM pour éviter les débordements lors du traitement de gros volumes.
    Utilise psutil pour monitorer et gc pour forcer la libération.
    """
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.process = psutil.Process(os.getpid())

    def get_current_usage_percent(self) -> float:
        return psutil.virtual_memory().percent

    def check_memory(self, stage: str = "unknown"):
        usage = self.get_current_usage_percent()
        if usage > self.config.max_ram_percent:
            print(f"[MEMORY WARNING] Stage {stage}: Usage at {usage}% (Limit: {self.config.max_ram_percent}%)")
            self.force_cleanup()

    def force_cleanup(self):
        """Force le garbage collector et tente de libérer la mémoire."""
        gc.collect()

    def get_safe_row_count(self, total_rows: int) -> int:
        """Calcule un nombre de lignes sûr à charger en fonction de safe_frac."""
        return int(total_rows * self.config.safe_frac)

    def log_resource_stats(self):
        mem_info = self.process.memory_info()
        rss_mb = mem_info.rss / (1024 * 1024)
        print(f"[RESOURCES] RAM RSS: {rss_mb:.2f} MB | CPU: {psutil.cpu_percent()}%")
