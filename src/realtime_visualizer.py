#!/usr/bin/env python3
"""
Real-time visualization module for pipeline progress and algorithm training
Provides algorithm-specific visualizations with real-time updates via callbacks
"""
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import time
import threading
from collections import defaultdict, deque
import warnings

warnings.filterwarnings('ignore')

# Setup logger
logger = logging.getLogger(__name__)

# Try to import matplotlib
try:
    import matplotlib
    # Use TkAgg backend if available for interactive plots, otherwise use Agg
    try:
        import tkinter
        matplotlib.use('TkAgg')
        INTERACTIVE_MODE = True
    except ImportError:
        matplotlib.use('Agg')
        INTERACTIVE_MODE = False
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
except ImportError:
    VISUALIZATION_AVAILABLE = False
    INTERACTIVE_MODE = False
    logger.warning("Matplotlib not available. Real-time visualizations disabled.")


class AlgorithmVisualizer:
    """Visualizer adapted for specific algorithm types"""
    
    def __init__(self, algorithm_name: str, output_dir: Path):
        """
        Initialize algorithm-specific visualizer
        
        Args:
            algorithm_name: Name of the algorithm (Logistic Regression, CNN, etc.)
            output_dir: Directory to save visualizations
        """
        self.algorithm_name = algorithm_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store metrics history
        self.metrics_history = defaultdict(list)
        self.timestamps = deque(maxlen=1000)  # Keep last 1000 points
        
        # Algorithm-specific settings
        self._setup_algorithm_config()
    
    def _setup_algorithm_config(self):
        """Setup algorithm-specific visualization configuration"""
        algo_lower = self.algorithm_name.lower()
        
        if 'logistic' in algo_lower or 'linear' in algo_lower:
            self.algo_type = 'linear'
            self.color_scheme = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            self.metrics_to_track = ['accuracy', 'f1_score', 'precision', 'recall', 'loss']
            
        elif 'tree' in algo_lower:
            self.algo_type = 'tree'
            self.color_scheme = ['#2ca02c', '#1f77b4', '#9467bd', '#8c564b']
            self.metrics_to_track = ['accuracy', 'f1_score', 'precision', 'recall', 'feature_importance']
            
        elif 'forest' in algo_lower:
            self.algo_type = 'ensemble'
            self.color_scheme = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
            self.metrics_to_track = ['accuracy', 'f1_score', 'precision', 'recall', 'feature_importance', 'n_estimators']
            
        elif 'cnn' in algo_lower or 'neural' in algo_lower:
            self.algo_type = 'deep_learning'
            self.color_scheme = ['#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']
            self.metrics_to_track = ['loss', 'val_loss', 'accuracy', 'val_accuracy', 'epoch']
            
        elif 'tabnet' in algo_lower:
            self.algo_type = 'deep_learning'
            self.color_scheme = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            self.metrics_to_track = ['loss', 'val_loss', 'accuracy', 'val_accuracy', 'epoch', 'sparsity']
            
        else:
            self.algo_type = 'generic'
            self.color_scheme = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            self.metrics_to_track = ['accuracy', 'f1_score', 'precision', 'recall']
    
    def update_metrics(self, metrics: Dict[str, float], timestamp: Optional[float] = None):
        """
        Update metrics for real-time tracking
        
        Args:
            metrics: Dictionary of metric_name -> value
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.timestamps.append(timestamp)
        
        for metric_name, value in metrics.items():
            if metric_name in self.metrics_to_track or metric_name in ['training_time', 'memory', 'fold']:
                self.metrics_history[metric_name].append(value)
    
    def create_training_visualization(self, save_path: Optional[Path] = None, show: bool = False) -> Optional[plt.Figure]:
        """
        Create algorithm-specific training visualization
        
        Args:
            save_path: Optional path to save figure
            show: Whether to display the figure (requires interactive backend)
            
        Returns:
            Matplotlib figure or None if visualization unavailable
        """
        if not VISUALIZATION_AVAILABLE or len(self.timestamps) == 0:
            return None
        
        try:
            fig = plt.figure(figsize=(14, 8))
            fig.suptitle(f'{self.algorithm_name} - Training Progress', fontsize=16, fontweight='bold')
            
            # Create subplots based on algorithm type
            if self.algo_type == 'deep_learning':
                # For deep learning: show loss/accuracy over epochs
                gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
                
                # Loss plot
                ax1 = fig.add_subplot(gs[0, 0])
                if 'loss' in self.metrics_history:
                    epochs = range(1, len(self.metrics_history['loss']) + 1)
                    ax1.plot(epochs, self.metrics_history['loss'], 
                            label='Training Loss', color=self.color_scheme[0], linewidth=2)
                    if 'val_loss' in self.metrics_history:
                        ax1.plot(epochs, self.metrics_history['val_loss'], 
                                label='Validation Loss', color=self.color_scheme[1], linewidth=2, linestyle='--')
                ax1.set_xlabel('Epoch', fontsize=10)
                ax1.set_ylabel('Loss', fontsize=10)
                ax1.set_title('Loss Evolution', fontsize=12, fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Accuracy plot
                ax2 = fig.add_subplot(gs[0, 1])
                if 'accuracy' in self.metrics_history:
                    epochs = range(1, len(self.metrics_history['accuracy']) + 1)
                    ax2.plot(epochs, self.metrics_history['accuracy'], 
                            label='Training Accuracy', color=self.color_scheme[2], linewidth=2)
                    if 'val_accuracy' in self.metrics_history:
                        ax2.plot(epochs, self.metrics_history['val_accuracy'], 
                                label='Validation Accuracy', color=self.color_scheme[3], linewidth=2, linestyle='--')
                ax2.set_xlabel('Epoch', fontsize=10)
                ax2.set_ylabel('Accuracy', fontsize=10)
                ax2.set_title('Accuracy Evolution', fontsize=12, fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim([0, 1.1])
                
                # Metrics comparison
                ax3 = fig.add_subplot(gs[1, :])
                metrics_to_show = ['f1_score', 'precision', 'recall']
                available_metrics = [m for m in metrics_to_show if m in self.metrics_history and len(self.metrics_history[m]) > 0]
                if available_metrics:
                    x = range(1, len(self.metrics_history[available_metrics[0]]) + 1)
                    for i, metric in enumerate(available_metrics):
                        ax3.plot(x, self.metrics_history[metric], 
                                label=metric.replace('_', ' ').title(), 
                                color=self.color_scheme[i % len(self.color_scheme)], linewidth=2)
                    ax3.set_xlabel('Epoch', fontsize=10)
                    ax3.set_ylabel('Score', fontsize=10)
                    ax3.set_title('Performance Metrics', fontsize=12, fontweight='bold')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                    ax3.set_ylim([0, 1.1])
            
            elif self.algo_type in ['tree', 'ensemble']:
                # For tree-based: show performance and feature importance
                gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
                
                # Performance metrics
                ax1 = fig.add_subplot(gs[0, :])
                metrics_to_show = ['accuracy', 'f1_score', 'precision', 'recall']
                available_metrics = [m for m in metrics_to_show if m in self.metrics_history and len(self.metrics_history[m]) > 0]
                if available_metrics:
                    x = range(1, len(self.metrics_history[available_metrics[0]]) + 1)
                    for i, metric in enumerate(available_metrics):
                        ax1.plot(x, self.metrics_history[metric], 
                                label=metric.replace('_', ' ').title(), 
                                color=self.color_scheme[i % len(self.color_scheme)], 
                                marker='o', markersize=4, linewidth=2)
                    ax1.set_xlabel('Fold' if 'fold' in str(self.metrics_history).lower() else 'Iteration', fontsize=10)
                    ax1.set_ylabel('Score', fontsize=10)
                    ax1.set_title('Cross-Validation Performance', fontsize=12, fontweight='bold')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    ax1.set_ylim([0, 1.1])
                
                # Feature importance (if available)
                if 'feature_importance' in self.metrics_history and len(self.metrics_history['feature_importance']) > 0:
                    ax2 = fig.add_subplot(gs[1, :])
                    # This would need to be implemented based on actual feature importance data
                    ax2.text(0.5, 0.5, 'Feature Importance\n(To be implemented)', 
                            ha='center', va='center', fontsize=12, transform=ax2.transAxes)
                    ax2.set_title('Feature Importance', fontsize=12, fontweight='bold')
            
            else:
                # Generic visualization for linear/other algorithms
                gs = fig.add_gridspec(2, 1, hspace=0.3)
                
                # Performance metrics
                ax1 = fig.add_subplot(gs[0])
                metrics_to_show = ['accuracy', 'f1_score', 'precision', 'recall']
                available_metrics = [m for m in metrics_to_show if m in self.metrics_history and len(self.metrics_history[m]) > 0]
                if available_metrics:
                    x = range(1, len(self.metrics_history[available_metrics[0]]) + 1)
                    for i, metric in enumerate(available_metrics):
                        ax1.plot(x, self.metrics_history[metric], 
                                label=metric.replace('_', ' ').title(), 
                                color=self.color_scheme[i % len(self.color_scheme)], 
                                marker='s', markersize=5, linewidth=2)
                    ax1.set_xlabel('Fold/Iteration', fontsize=10)
                    ax1.set_ylabel('Score', fontsize=10)
                    ax1.set_title('Performance Metrics', fontsize=12, fontweight='bold')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    ax1.set_ylim([0, 1.1])
                
                # Resource usage
                ax2 = fig.add_subplot(gs[1])
                if 'training_time' in self.metrics_history or 'memory' in self.metrics_history:
                    if 'training_time' in self.metrics_history:
                        ax2_twin = ax2.twinx()
                        time_data = self.metrics_history['training_time']
                        ax2_twin.plot(range(1, len(time_data) + 1), time_data, 
                                     color=self.color_scheme[0], marker='o', label='Training Time (s)')
                        ax2_twin.set_ylabel('Time (seconds)', color=self.color_scheme[0], fontsize=10)
                        ax2_twin.tick_params(axis='y', labelcolor=self.color_scheme[0])
                        ax2_twin.legend(loc='upper right')
                    if 'memory' in self.metrics_history:
                        mem_data = self.metrics_history['memory']
                        ax2.plot(range(1, len(mem_data) + 1), mem_data, 
                                color=self.color_scheme[1], marker='s', label='Memory (MB)')
                        ax2.set_ylabel('Memory (MB)', color=self.color_scheme[1], fontsize=10)
                        ax2.tick_params(axis='y', labelcolor=self.color_scheme[1])
                        ax2.legend(loc='upper left')
                    ax2.set_xlabel('Fold/Iteration', fontsize=10)
                    ax2.set_title('Resource Usage', fontsize=12, fontweight='bold')
                    ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved visualization to {save_path}")
            
            if show and INTERACTIVE_MODE:
                plt.show(block=False)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating visualization for {self.algorithm_name}: {e}", exc_info=True)
            return None


class RealTimeVisualizer:
    """Main real-time visualizer for pipeline progress"""
    
    def __init__(self, output_dir: Path, enable_realtime: bool = True):
        """
        Initialize real-time visualizer
        
        Args:
            output_dir: Output directory for saved visualizations
            enable_realtime: Whether to enable real-time interactive plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_realtime = enable_realtime and INTERACTIVE_MODE
        
        # Store algorithm visualizers
        self.algorithm_visualizers: Dict[str, AlgorithmVisualizer] = {}
        
        # Pipeline-level metrics
        self.pipeline_metrics = {
            'data_loading': defaultdict(list),
            'preprocessing': defaultdict(list),
            'training': defaultdict(list),
            'evaluation': defaultdict(list)
        }
        
        # Thread-safe lock for updates
        self.lock = threading.Lock()
        
        # Active figures (for real-time updates)
        self.active_figures: Dict[str, plt.Figure] = {}
        
        # Pipeline overview figure (for real-time updates)
        self.pipeline_overview_fig: Optional[plt.Figure] = None
        self.pipeline_overview_axes: Optional[np.ndarray] = None
        self.pipeline_overview_update_count = 0
        
        logger.info(f"RealTimeVisualizer initialized (interactive: {self.enable_realtime})")
    
    def get_algorithm_visualizer(self, algorithm_name: str) -> AlgorithmVisualizer:
        """Get or create algorithm-specific visualizer"""
        if algorithm_name not in self.algorithm_visualizers:
            algo_output_dir = self.output_dir / 'algorithm_visualizations'
            self.algorithm_visualizers[algorithm_name] = AlgorithmVisualizer(
                algorithm_name, algo_output_dir
            )
        return self.algorithm_visualizers[algorithm_name]
    
    def callback_data_loading(self, data: Dict[str, Any]):
        """Callback for data loading progress"""
        with self.lock:
            if data.get('type') == 'loading':
                dataset = data.get('dataset', 'unknown')
                chunks = data.get('chunks_processed', 0)
                rows = data.get('rows_loaded', 0)
                
                self.pipeline_metrics['data_loading'][dataset].append({
                    'chunks': chunks,
                    'rows': rows,
                    'timestamp': time.time()
                })
                
                logger.debug(f"[VIZ] Data loading update: {dataset} - {rows:,} rows")
    
    def callback_training_progress(self, data: Dict[str, Any]):
        """Callback for algorithm training progress"""
        with self.lock:
            algorithm_name = data.get('algorithm', 'unknown')
            metrics = data.get('metrics', {})
            fold = data.get('fold', None)
            
            viz = self.get_algorithm_visualizer(algorithm_name)
            viz.update_metrics(metrics)
            
            if fold:
                logger.debug(f"[VIZ] Training update: {algorithm_name} - Fold {fold} - {metrics}")
            
            # Auto-save visualization every 5 updates
            if len(viz.timestamps) % 5 == 0:
                self._update_algorithm_visualization(algorithm_name, viz)
                # Update pipeline overview in real-time
                self._update_pipeline_overview_realtime()
    
    def callback_evaluation_progress(self, data: Dict[str, Any]):
        """Callback for evaluation progress"""
        with self.lock:
            algorithm_name = data.get('algorithm', 'unknown')
            metrics = data.get('metrics', {})
            
            viz = self.get_algorithm_visualizer(algorithm_name)
            viz.update_metrics(metrics)
            
            logger.debug(f"[VIZ] Evaluation update: {algorithm_name} - {metrics}")
    
    def _update_algorithm_visualization(self, algorithm_name: str, viz: AlgorithmVisualizer):
        """Update and save algorithm visualization"""
        try:
            save_path = viz.output_dir / f"{algorithm_name.replace(' ', '_')}_realtime.png"
            fig = viz.create_training_visualization(save_path=save_path, show=False)
            
            if fig and self.enable_realtime:
                # Update existing figure or create new
                if algorithm_name in self.active_figures:
                    plt.close(self.active_figures[algorithm_name])
                self.active_figures[algorithm_name] = fig
                plt.pause(0.01)  # Small pause to allow update
            
        except Exception as e:
            logger.warning(f"Could not update visualization for {algorithm_name}: {e}")
    
    def _update_pipeline_overview_realtime(self):
        """Update pipeline overview visualization in real-time"""
        try:
            # Only update every 3 algorithm updates to avoid too frequent refreshes
            self.pipeline_overview_update_count += 1
            if self.pipeline_overview_update_count % 3 != 0:
                return
            
            if not VISUALIZATION_AVAILABLE:
                return
            
            # Create or update pipeline overview
            if self.pipeline_overview_fig is None or self.pipeline_overview_axes is None:
                # Create new figure
                self.pipeline_overview_fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                self.pipeline_overview_axes = axes
                self.pipeline_overview_fig.suptitle('Pipeline Progress Overview (Live)', 
                                                    fontsize=16, fontweight='bold')
                plt.ion()  # Turn on interactive mode
                if self.enable_realtime:
                    plt.show(block=False)
            else:
                # Clear existing axes
                axes = self.pipeline_overview_axes
                for ax in axes.flat:
                    ax.clear()
            
            # Data loading progress
            ax1 = axes[0, 0]
            for dataset, data_list in self.pipeline_metrics['data_loading'].items():
                if data_list:
                    rows = [d['rows'] for d in data_list]
                    chunks = [d['chunks'] for d in data_list]
                    ax1.plot(chunks, rows, marker='o', label=dataset, linewidth=2)
            ax1.set_xlabel('Chunks Processed')
            ax1.set_ylabel('Rows Loaded')
            ax1.set_title('Data Loading Progress')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Algorithm performance comparison
            ax2 = axes[0, 1]
            algo_names = []
            final_accuracies = []
            for algo_name, viz in self.algorithm_visualizers.items():
                if 'accuracy' in viz.metrics_history and len(viz.metrics_history['accuracy']) > 0:
                    algo_names.append(algo_name)
                    final_accuracies.append(viz.metrics_history['accuracy'][-1])
            
            if algo_names:
                colors = plt.cm.viridis(np.linspace(0, 1, len(algo_names)))
                ax2.barh(algo_names, final_accuracies, color=colors)
                ax2.set_xlabel('Final Accuracy')
                ax2.set_title('Algorithm Performance Comparison')
                ax2.set_xlim([0, 1.1])
                ax2.grid(True, alpha=0.3, axis='x')
            else:
                ax2.text(0.5, 0.5, 'No algorithms evaluated yet', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Algorithm Performance Comparison')
            
            # Training time comparison
            ax3 = axes[1, 0]
            algo_names = []
            training_times = []
            for algo_name, viz in self.algorithm_visualizers.items():
                if 'training_time' in viz.metrics_history and len(viz.metrics_history['training_time']) > 0:
                    algo_names.append(algo_name)
                    training_times.append(np.mean(viz.metrics_history['training_time']))
            
            if algo_names:
                colors = plt.cm.plasma(np.linspace(0, 1, len(algo_names)))
                ax3.barh(algo_names, training_times, color=colors)
                ax3.set_xlabel('Average Training Time (s)')
                ax3.set_title('Training Time Comparison')
                ax3.grid(True, alpha=0.3, axis='x')
            else:
                ax3.text(0.5, 0.5, 'No training data yet', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                ax3.set_title('Training Time Comparison')
            
            # F1 Score comparison
            ax4 = axes[1, 1]
            algo_names = []
            f1_scores = []
            for algo_name, viz in self.algorithm_visualizers.items():
                if 'f1_score' in viz.metrics_history and len(viz.metrics_history['f1_score']) > 0:
                    algo_names.append(algo_name)
                    f1_scores.append(viz.metrics_history['f1_score'][-1])
            
            if algo_names:
                colors = plt.cm.coolwarm(np.linspace(0, 1, len(algo_names)))
                ax4.barh(algo_names, f1_scores, color=colors)
                ax4.set_xlabel('F1 Score')
                ax4.set_title('F1 Score Comparison')
                ax4.set_xlim([0, 1.1])
                ax4.grid(True, alpha=0.3, axis='x')
            else:
                ax4.text(0.5, 0.5, 'No F1 scores yet', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.set_title('F1 Score Comparison')
            
            plt.tight_layout()
            
            # Update figure
            self.pipeline_overview_fig.canvas.draw()
            self.pipeline_overview_fig.canvas.flush_events()
            
            # Save snapshot periodically (every 10 updates)
            if self.pipeline_overview_update_count % 10 == 0:
                save_path = self.output_dir / 'pipeline_overview_realtime.png'
                self.pipeline_overview_fig.savefig(save_path, dpi=150, bbox_inches='tight')
            
        except Exception as e:
            logger.debug(f"Error updating pipeline overview in real-time: {e}")
    
    def finalize_algorithm_visualizations(self):
        """Finalize and save all algorithm visualizations"""
        logger.info("[VIZ] Finalizing algorithm visualizations...")
        
        for algorithm_name, viz in self.algorithm_visualizers.items():
            try:
                save_path = viz.output_dir / f"{algorithm_name.replace(' ', '_')}_final.png"
                fig = viz.create_training_visualization(save_path=save_path, show=False)
                
                if fig:
                    logger.info(f"  ✓ Saved final visualization for {algorithm_name}")
                    plt.close(fig)
                    
            except Exception as e:
                logger.warning(f"Could not finalize visualization for {algorithm_name}: {e}")
        
        # Close all active figures
        for fig in self.active_figures.values():
            try:
                plt.close(fig)
            except:
                pass
        self.active_figures.clear()
        
        # Close pipeline overview figure
        if self.pipeline_overview_fig:
            try:
                plt.close(self.pipeline_overview_fig)
            except:
                pass
            self.pipeline_overview_fig = None
            self.pipeline_overview_axes = None
    
    def create_pipeline_overview(self, save_path: Optional[Path] = None):
        """Create overview visualization of entire pipeline"""
        if not VISUALIZATION_AVAILABLE:
            return None
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Pipeline Progress Overview', fontsize=16, fontweight='bold')
            
            # Data loading progress
            ax1 = axes[0, 0]
            for dataset, data_list in self.pipeline_metrics['data_loading'].items():
                if data_list:
                    rows = [d['rows'] for d in data_list]
                    chunks = [d['chunks'] for d in data_list]
                    ax1.plot(chunks, rows, marker='o', label=dataset, linewidth=2)
            ax1.set_xlabel('Chunks Processed')
            ax1.set_ylabel('Rows Loaded')
            ax1.set_title('Data Loading Progress')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Algorithm performance comparison
            ax2 = axes[0, 1]
            algo_names = []
            final_accuracies = []
            for algo_name, viz in self.algorithm_visualizers.items():
                if 'accuracy' in viz.metrics_history and len(viz.metrics_history['accuracy']) > 0:
                    algo_names.append(algo_name)
                    final_accuracies.append(viz.metrics_history['accuracy'][-1])
            
            if algo_names:
                colors = plt.cm.viridis(np.linspace(0, 1, len(algo_names)))
                ax2.barh(algo_names, final_accuracies, color=colors)
                ax2.set_xlabel('Final Accuracy')
                ax2.set_title('Algorithm Performance Comparison')
                ax2.set_xlim([0, 1.1])
                ax2.grid(True, alpha=0.3, axis='x')
            
            # Training time comparison
            ax3 = axes[1, 0]
            algo_names = []
            training_times = []
            for algo_name, viz in self.algorithm_visualizers.items():
                if 'training_time' in viz.metrics_history and len(viz.metrics_history['training_time']) > 0:
                    algo_names.append(algo_name)
                    training_times.append(np.mean(viz.metrics_history['training_time']))
            
            if algo_names:
                colors = plt.cm.plasma(np.linspace(0, 1, len(algo_names)))
                ax3.barh(algo_names, training_times, color=colors)
                ax3.set_xlabel('Average Training Time (s)')
                ax3.set_title('Training Time Comparison')
                ax3.grid(True, alpha=0.3, axis='x')
            
            # F1 Score comparison
            ax4 = axes[1, 1]
            algo_names = []
            f1_scores = []
            for algo_name, viz in self.algorithm_visualizers.items():
                if 'f1_score' in viz.metrics_history and len(viz.metrics_history['f1_score']) > 0:
                    algo_names.append(algo_name)
                    f1_scores.append(viz.metrics_history['f1_score'][-1])
            
            if algo_names:
                colors = plt.cm.coolwarm(np.linspace(0, 1, len(algo_names)))
                ax4.barh(algo_names, f1_scores, color=colors)
                ax4.set_xlabel('F1 Score')
                ax4.set_title('F1 Score Comparison')
                ax4.set_xlim([0, 1.1])
                ax4.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = self.output_dir / 'pipeline_overview.png'
            
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved pipeline overview to {save_path}")
            plt.close(fig)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating pipeline overview: {e}", exc_info=True)
            return None


def create_realtime_callback(visualizer: RealTimeVisualizer) -> Callable:
    """
    Create a callback function for dataset loader that integrates with visualizer
    
    Args:
        visualizer: RealTimeVisualizer instance
        
    Returns:
        Callback function
    """
    def callback(data: Dict[str, Any]):
        try:
            visualizer.callback_data_loading(data)
        except Exception as e:
            logger.warning(f"Error in visualization callback: {e}")
    
    return callback
