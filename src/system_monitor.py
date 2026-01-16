#!/usr/bin/env python3
"""
System monitoring and resource management
Monitors RAM, CPU, and calculates ETA for pipeline steps
"""
import psutil
import time
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitor system resources and calculate ETA"""
    
    def __init__(self, max_memory_percent: float = 90.0):
        """
        Initialize system monitor
        
        Args:
            max_memory_percent: Maximum memory usage percentage before warnings (default: 90%)
        """
        self.max_memory_percent = max_memory_percent
        self.process = psutil.Process()
        self.start_time = None
        self.progress_history = []  # List of (time, progress) tuples
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        Get current memory information
        
        Returns:
            Dictionary with memory stats in GB and percentage
        """
        mem = psutil.virtual_memory()
        process_mem = self.process.memory_info().rss / (1024**3)  # GB
        
        return {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'used_gb': mem.used / (1024**3),
            'used_percent': mem.percent,
            'process_mem_gb': process_mem,
            'process_mem_mb': process_mem * 1024
        }
    
    def get_cpu_info(self) -> Dict[str, float]:
        """
        Get current CPU information
        
        Returns:
            Dictionary with CPU stats
        """
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        
        return {
            'percent': cpu_percent,
            'count': cpu_count
        }
    
    def calculate_optimal_chunk_size(self, estimated_row_size_bytes: int = 500) -> int:
        """
        Calculate optimal chunk size based on available memory
        
        Args:
            estimated_row_size_bytes: Estimated size of one row in bytes (default: 500 bytes)
            
        Returns:
            Optimal chunk size in rows
        """
        mem_info = self.get_memory_info()
        available_mb = mem_info['available_gb'] * 1024
        
        # Use only up to 50% of available memory for chunks (leave room for processing)
        safe_memory_mb = available_mb * 0.5
        
        # Calculate how many rows we can fit in safe memory
        # Each chunk is loaded multiple times (for pandas operations), so be conservative
        rows_per_mb = (1024 * 1024) / estimated_row_size_bytes
        max_rows = int(safe_memory_mb * rows_per_mb / 2)  # Divide by 2 for safety
        
        # Clamp between reasonable values
        optimal_chunk = max(100_000, min(max_rows, 10_000_000))
        
        logger.debug(f"[MONITOR] Available memory: {available_mb:.1f} MB, Optimal chunk size: {optimal_chunk:,} rows")
        return optimal_chunk
    
    def check_memory_safe(self) -> Tuple[bool, str]:
        """
        Check if memory usage is safe (< max_memory_percent)
        
        Returns:
            Tuple of (is_safe, message)
        """
        mem_info = self.get_memory_info()
        
        if mem_info['used_percent'] >= self.max_memory_percent:
            return False, f"Memory usage ({mem_info['used_percent']:.1f}%) exceeds threshold ({self.max_memory_percent}%)"
        elif mem_info['used_percent'] >= self.max_memory_percent * 0.8:
            return True, f"Warning: Memory usage is high ({mem_info['used_percent']:.1f}%)"
        else:
            return True, f"Memory usage: {mem_info['used_percent']:.1f}%"
    
    def start_progress_tracking(self):
        """Start tracking progress for ETA calculation"""
        self.start_time = time.time()
        self.progress_history = [(time.time(), 0.0)]
    
    def update_progress(self, current: int, total: int, item_name: str = "items") -> Dict[str, any]:
        """
        Update progress and calculate ETA
        
        Args:
            current: Current progress (number of items processed)
            total: Total items to process
            item_name: Name of items being processed
            
        Returns:
            Dictionary with progress info and ETA
        """
        if total == 0:
            return {
                'percent': 0.0,
                'current': 0,
                'total': 0,
                'eta_seconds': None,
                'eta_formatted': 'N/A',
                'elapsed_seconds': 0,
                'elapsed_formatted': '0:00:00',
                'items_per_second': 0.0
            }
        
        progress = current / total
        current_time = time.time()
        
        # Add to history
        self.progress_history.append((current_time, progress))
        
        # Keep only last 10 points for ETA calculation
        if len(self.progress_history) > 10:
            self.progress_history = self.progress_history[-10:]
        
        # Calculate elapsed time
        if self.start_time is None:
            self.start_time = current_time
        
        elapsed = current_time - self.start_time
        
        # Calculate ETA based on recent progress rate
        if len(self.progress_history) >= 2 and elapsed > 5:  # Need at least 5 seconds of data
            recent_progress = self.progress_history[-1][1] - self.progress_history[-5][1] if len(self.progress_history) >= 5 else progress
            recent_time = self.progress_history[-1][0] - self.progress_history[-5][0] if len(self.progress_history) >= 5 else elapsed
            
            if recent_time > 0 and recent_progress < 1.0:
                rate = recent_progress / recent_time  # progress per second
                remaining_progress = 1.0 - progress
                eta_seconds = remaining_progress / rate if rate > 0 else None
            else:
                eta_seconds = None
        else:
            # Use average rate so far
            if elapsed > 0 and progress > 0:
                rate = progress / elapsed
                remaining_progress = 1.0 - progress
                eta_seconds = remaining_progress / rate if rate > 0 else None
            else:
                eta_seconds = None
        
        # Calculate items per second
        items_per_second = current / elapsed if elapsed > 0 else 0.0
        
        return {
            'percent': progress * 100,
            'current': current,
            'total': total,
            'eta_seconds': eta_seconds,
            'eta_formatted': str(timedelta(seconds=int(eta_seconds))) if eta_seconds else 'Calculating...',
            'elapsed_seconds': elapsed,
            'elapsed_formatted': str(timedelta(seconds=int(elapsed))),
            'items_per_second': items_per_second,
            'item_name': item_name
        }
    
    def get_system_summary(self) -> str:
        """Get formatted system summary string"""
        mem_info = self.get_memory_info()
        cpu_info = self.get_cpu_info()
        
        summary = (
            f"System: RAM {mem_info['used_percent']:.1f}% used "
            f"({mem_info['used_gb']:.1f}/{mem_info['total_gb']:.1f} GB), "
            f"CPU {cpu_info['percent']:.1f}%, "
            f"Process RAM: {mem_info['process_mem_gb']:.2f} GB"
        )
        return summary
