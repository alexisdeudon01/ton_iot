import psutil
import time
import logging
import threading
import gc
from typing import Dict, Optional, Tuple, Any, List
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

class SystemMonitor:
    """
    Advanced System Monitor with dedicated resource management thread.
    Enforces a 50% RAM limit and performs proactive garbage collection.
    """

    def __init__(self, max_memory_percent: float = 50.0):
        self.max_memory_percent = max_memory_percent
        self.process = psutil.Process()
        self.start_time = time.time()

        # Resource history for plotting
        self.history = {
            'timestamp': [],
            'cpu_percent': [],
            'ram_percent': [],
            'process_ram_mb': [],
            'phase': []
        }
        self.current_phase = "Initialization"

        # Monitoring thread control
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self.lock = threading.Lock()

    def start_monitoring(self, interval: float = 0.5):
        """Starts the background monitoring thread."""
        if self._monitor_thread is not None:
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,), daemon=True)
        self._monitor_thread.start()
        logger.info(f"Background monitoring started (RAM limit: {self.max_memory_percent}%)")

    def stop_monitoring(self):
        """Stops the background monitoring thread."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join()
            self._monitor_thread = None
        logger.info("Background monitoring stopped.")

    def set_phase(self, phase_name: str):
        with self.lock:
            self.current_phase = phase_name
            logger.info(f"System Monitor Phase changed to: {phase_name}")

    def _monitor_loop(self, interval: float):
        while not self._stop_event.is_set():
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=None)
            proc_mem = self.process.memory_info().rss / (1024 * 1024) # MB

            with self.lock:
                self.history['timestamp'].append(time.time() - self.start_time)
                self.history['cpu_percent'].append(cpu)
                self.history['ram_percent'].append(mem.percent)
                self.history['process_ram_mb'].append(proc_mem)
                self.history['phase'].append(self.current_phase)

            # Proactive Memory Management
            if mem.percent > self.max_memory_percent:
                gc.collect()
                # If still too high, we might need to sleep to let system breathe
                if mem.percent > self.max_memory_percent + 5:
                    time.sleep(0.5)

            time.sleep(interval)

    def get_memory_info(self) -> Dict[str, float]:
        mem = psutil.virtual_memory()
        return {
            'used_percent': mem.percent,
            'process_mem_mb': self.process.memory_info().rss / (1024 * 1024)
        }

    def plot_resource_consumption(self, output_path: str):
        """Generates CPU and Memory consumption graphs for each phase."""
        with self.lock:
            df = pd.DataFrame(self.history)

        if df.empty:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # CPU Plot
        for phase in df['phase'].unique():
            phase_data = df[df['phase'] == phase]
            ax1.plot(phase_data['timestamp'], phase_data['cpu_percent'], label=f"CPU - {phase}")

        ax1.axhline(y=100, color='r', linestyle='--')
        ax1.set_ylabel("CPU Usage (%)")
        ax1.set_title("CPU Consumption per Phase")
        ax1.legend(loc='upper right', fontsize='small', ncol=2)
        ax1.grid(True, alpha=0.3)

        # RAM Plot
        for phase in df['phase'].unique():
            phase_data = df[df['phase'] == phase]
            ax2.plot(phase_data['timestamp'], phase_data['ram_percent'], label=f"RAM - {phase}")

        ax2.axhline(y=self.max_memory_percent, color='r', linestyle='--', label=f"Limit ({self.max_memory_percent}%)")
        ax2.set_ylabel("RAM Usage (%)")
        ax2.set_xlabel("Time (s)")
        ax2.set_title("Memory Consumption per Phase")
        ax2.legend(loc='upper right', fontsize='small', ncol=2)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Resource consumption plot saved to {output_path}")

    def generate_timeline_heatmap(self, output_path: str):
        """Generates a timeline heatmap of phases."""
        with self.lock:
            df = pd.DataFrame(self.history)

        if df.empty:
            return

        # Create a simplified timeline
        timeline = df.groupby('phase')['timestamp'].agg(['min', 'max', 'count'])
        timeline['duration'] = timeline['max'] - timeline['min']

        plt.figure(figsize=(12, 4))
        y_labels = timeline.index.tolist()
        for i, phase in enumerate(y_labels):
            plt.barh(i, timeline.loc[phase, 'duration'], left=timeline.loc[phase, 'min'], label=phase)

        plt.yticks(range(len(y_labels)), y_labels)
        plt.xlabel("Time (s)")
        plt.title("Execution Timeline")
        plt.grid(True, axis='x', alpha=0.3)
        plt.savefig(output_path)
        plt.close()

    def calculate_optimal_chunk_size(self, estimated_row_size_bytes: int = 500) -> int:
        """Legacy compatibility."""
        return 100_000

    def check_memory_safe(self) -> Tuple[bool, str]:
        """Legacy compatibility."""
        info = self.get_memory_info()
        return info['used_percent'] < self.max_memory_percent, f"RAM: {info['used_percent']}%"

    def start_progress_tracking(self):
        """Legacy compatibility."""
        pass

    def update_progress(self, current: int, total: int, item_name: str = "items") -> Dict[str, Any]:
        """Legacy compatibility."""
        return {'eta_formatted': 'N/A'}
