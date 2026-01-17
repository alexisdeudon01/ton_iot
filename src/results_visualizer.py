#!/usr/bin/env python3
"""
Results Visualizer with Tkinter GUI
Displays detailed evaluation results, confusion matrices, and performance metrics
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('TkAgg')  # Use TkAgg backend for Tkinter
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available. Visualization disabled.")


class ResultsVisualizer:
    """Tkinter GUI for visualizing IRP pipeline results"""

    def __init__(self, output_dir: Path):
        """
        Initialize the results visualizer

        Args:
            output_dir: Path to output directory containing results
        """
        self.output_dir = Path(output_dir)
        self.root = tk.Tk()
        self.root.title("IRP Research Pipeline - Results Visualizer")
        self.root.geometry("1400x900")

        # Data storage
        self.evaluation_results = None
        self.ranking_results = None
        self.preprocessing_stats = None

        # Load data
        self._load_data()

        # Create GUI
        self._create_gui()

    def _load_data(self):
        """Load all result files"""
        try:
            # Load evaluation results
            eval_path = self.output_dir / 'phase3_evaluation' / 'evaluation_results.csv'
            if eval_path.exists():
                self.evaluation_results = pd.read_csv(eval_path)
                logger.info(f"Loaded evaluation results: {len(self.evaluation_results)} models")

            # Load ranking results
            ranking_path = self.output_dir / 'phase5_ranking' / 'ranking_results.csv'
            if ranking_path.exists():
                self.ranking_results = pd.read_csv(ranking_path)
                logger.info(f"Loaded ranking results: {len(self.ranking_results)} models")

            # Load preprocessing stats if available
            preprocessed_path = self.output_dir / 'phase1_preprocessing' / 'preprocessed_data.csv'
            if preprocessed_path.exists():
                df = pd.read_csv(preprocessed_path)
                self.preprocessing_stats = {
                    'total_samples': len(df),
                    'total_features': len(df.columns) - 1,  # Exclude label
                    'class_distribution': df['label'].value_counts().to_dict() if 'label' in df.columns else {}
                }
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load results: {e}")

    def _create_gui(self):
        """Create the main GUI interface"""
        # Create notebook (tabs)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Overview
        self._create_overview_tab(notebook)

        # Tab 2: Evaluation Details
        if self.evaluation_results is not None:
            self._create_evaluation_tab(notebook)

        # Tab 3: Ranking
        if self.ranking_results is not None:
            self._create_ranking_tab(notebook)

        # Tab 4: Preprocessing Stats
        if self.preprocessing_stats is not None:
            self._create_preprocessing_tab(notebook)

        # Tab 5: Matrices Details
        self._create_matrices_tab(notebook)

    def _create_overview_tab(self, notebook):
        """Create overview tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Overview")

        # Title
        title = tk.Label(frame, text="IRP Research Pipeline - Results Overview",
                        font=("Arial", 16, "bold"))
        title.pack(pady=10)

        # Summary text
        summary_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=30)
        summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        summary = "=== IRP RESEARCH PIPELINE RESULTS ===\n\n"

        if self.preprocessing_stats:
            summary += "📊 PREPROCESSING STATISTICS\n"
            summary += f"Total Samples: {self.preprocessing_stats['total_samples']:,}\n"
            summary += f"Total Features: {self.preprocessing_stats['total_features']}\n"
            if self.preprocessing_stats['class_distribution']:
                summary += "Class Distribution:\n"
                for label, count in self.preprocessing_stats['class_distribution'].items():
                    summary += f"  Class {label}: {count:,} ({count/self.preprocessing_stats['total_samples']*100:.1f}%)\n"
            summary += "\n"

        if self.evaluation_results is not None:
            summary += "📈 EVALUATION RESULTS\n"
            summary += f"Models Evaluated: {len(self.evaluation_results)}\n\n"

            # Best model per dimension
            if 'f1_score' in self.evaluation_results.columns:
                best_f1 = self.evaluation_results.loc[self.evaluation_results['f1_score'].idxmax()]
                summary += f"Best Detection Performance (F1): {best_f1['model_name']} (F1={best_f1['f1_score']:.4f})\n"

            if 'training_time_seconds' in self.evaluation_results.columns:
                best_time = self.evaluation_results.loc[self.evaluation_results['training_time_seconds'].idxmin()]
                summary += f"Best Resource Efficiency: {best_time['model_name']} (Time={best_time['training_time_seconds']:.2f}s)\n"

            if 'explainability_score' in self.evaluation_results.columns:
                best_exp = self.evaluation_results.loc[self.evaluation_results['explainability_score'].idxmax()]
                summary += f"Best Explainability: {best_exp['model_name']} (Score={best_exp['explainability_score']:.4f})\n"
            summary += "\n"

        if self.ranking_results is not None:
            summary += "🏆 FINAL RANKING (AHP-TOPSIS)\n"
            summary += f"Total Models Ranked: {len(self.ranking_results)}\n\n"
            if 'rank' in self.ranking_results.columns and 'alternative' in self.ranking_results.columns:
                top_3 = self.ranking_results.nsmallest(3, 'rank') if 'rank' in self.ranking_results.columns else self.ranking_results.head(3)
                for idx, row in top_3.iterrows():
                    summary += f"Rank {row['rank']}: {row['alternative']}\n"

        summary_text.insert(tk.END, summary)
        summary_text.config(state=tk.DISABLED)

    def _create_evaluation_tab(self, notebook):
        """Create detailed evaluation tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Evaluation Details")

        # Create treeview for table
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")

        # Treeview
        columns = list(self.evaluation_results.columns)
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings",
                           yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)

        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor=tk.CENTER)

        # Insert data
        for idx, row in self.evaluation_results.iterrows():
            values = [f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
                     for val in row.values]
            tree.insert("", tk.END, values=values)

        # Pack treeview and scrollbars
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        # Buttons frame
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=5)

        btn_details = tk.Button(btn_frame, text="Show Detailed Matrices",
                               command=self._show_matrix_details)
        btn_details.pack(side=tk.LEFT, padx=5)

        btn_viz = tk.Button(btn_frame, text="Show Algorithm Visualizations",
                           command=self._show_algorithm_visualizations_popup)
        btn_viz.pack(side=tk.LEFT, padx=5)

        # Double-click to show algorithm popup
        tree.bind('<Double-1>', lambda e: self._show_algorithm_popup(tree))

    def _create_ranking_tab(self, notebook):
        """Create ranking tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Ranking")

        # Create treeview
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")

        columns = list(self.ranking_results.columns)
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings",
                           yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, anchor=tk.CENTER)

        for idx, row in self.ranking_results.iterrows():
            values = [f"{val:.6f}" if isinstance(val, (int, float)) else str(val)
                     for val in row.values]
            tree.insert("", tk.END, values=values)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

    def _create_preprocessing_tab(self, notebook):
        """Create preprocessing statistics tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Preprocessing")

        text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=30)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        content = "=== PREPROCESSING STATISTICS ===\n\n"
        if self.preprocessing_stats:
            for key, value in self.preprocessing_stats.items():
                content += f"{key}: {value}\n\n"

        text.insert(tk.END, content)
        text.config(state=tk.DISABLED)

    def _create_matrices_tab(self, notebook):
        """Create detailed matrices tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Matrices Details")

        # Text area for matrix details
        text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=35)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        content = self._generate_matrix_details()
        text.insert(tk.END, content)
        text.config(state=tk.DISABLED)

    def _generate_matrix_details(self) -> str:
        """Generate detailed matrix information"""
        details = "=== DETAILED MATRICES INFORMATION ===\n\n"

        eval_results = self.evaluation_results
        if eval_results is not None:
            details += "📊 DETECTION PERFORMANCE MATRIX (Dimension 1)\n"
            details += "=" * 60 + "\n\n"

            # Performance metrics matrix
            perf_cols = ['model_name', 'f1_score', 'accuracy', 'precision', 'recall']
            available_perf = [col for col in perf_cols if col in eval_results.columns]

            if available_perf:
                perf_df = eval_results[available_perf].copy()
                details += "Performance Metrics Table:\n"
                details += perf_df.to_string(index=False)
                details += "\n\n"

                # Confusion Matrix Details for each model
                details += "🔍 CONFUSION MATRICES DETAILS:\n"
                details += "-" * 60 + "\n"

                # Try to load detailed results from algorithm reports or JSON if available
                for idx, row in perf_df.iterrows():
                    model_name = str(row['model_name'])
                    details += f"\n{model_name}:\n"
                    details += f"  F1 Score: {float(row['f1_score']):.4f}\n"
                    details += f"  Precision: {float(row['precision']):.4f}\n"
                    details += f"  Recall: {float(row['recall']):.4f}\n"
                    details += f"  Accuracy: {float(row['accuracy']):.4f}\n"

                details += "\n"

                # Statistical summary
                details += "Statistical Summary:\n"
                numeric_cols = perf_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    details += perf_df[numeric_cols].describe().to_string()
                    details += "\n\n"

            details += "📈 RESOURCE EFFICIENCY MATRIX (Dimension 2)\n"
            details += "=" * 60 + "\n\n"

            # Resource metrics matrix
            resource_cols = ['model_name', 'training_time_seconds', 'memory_used_mb', 'peak_memory_mb']
            available_resource = [col for col in resource_cols if col in self.evaluation_results.columns]

            if available_resource:
                resource_df = self.evaluation_results[available_resource].copy()
                details += resource_df.to_string(index=False)
                details += "\n\n"

                # Statistical summary
                numeric_cols = resource_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    details += "Statistical Summary:\n"
                    details += resource_df[numeric_cols].describe().to_string()
                    details += "\n\n"

            details += "🔍 EXPLAINABILITY MATRIX (Dimension 3)\n"
            details += "=" * 60 + "\n\n"

            # Explainability metrics matrix
            exp_cols = ['model_name', 'explainability_score', 'shap_score', 'lime_score', 'native_interpretability']
            available_exp = [col for col in exp_cols if col in self.evaluation_results.columns]

            if available_exp:
                exp_df = self.evaluation_results[available_exp].copy()
                details += exp_df.to_string(index=False)
                details += "\n\n"

                # Statistical summary
                numeric_cols = exp_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    details += "Statistical Summary:\n"
                    details += exp_df[numeric_cols].describe().to_string()
                    details += "\n\n"

        if self.ranking_results is not None:
            details += "🏆 RANKING MATRIX (AHP-TOPSIS)\n"
            details += "=" * 60 + "\n\n"
            details += self.ranking_results.to_string(index=False)
            details += "\n\n"

        return details

    def _show_matrix_details(self):
        """Show detailed matrix information in a new window"""
        window = tk.Toplevel(self.root)
        window.title("Detailed Matrices")
        window.geometry("1000x700")

        text = scrolledtext.ScrolledText(window, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        content = self._generate_matrix_details()
        text.insert(tk.END, content)
        text.config(state=tk.DISABLED)

    def _show_algorithm_popup(self, tree):
        """Show popup window for selected algorithm"""
        selection = tree.selection()
        if not selection:
            return

        item = tree.item(selection[0])
        values = item['values']
        if not values:
            return

        # Get model name from first column
        model_name = values[0] if values else "Unknown"

        logger.info(f"[ACTION] Showing popup for algorithm: {model_name}")
        self._create_algorithm_popup(model_name)

    def _show_algorithm_visualizations_popup(self):
        """Show popup with visualizations for all algorithms"""
        logger.info("[ACTION] Showing algorithm visualizations popup")
        popup = tk.Toplevel(self.root)
        popup.title("Algorithm Visualizations")
        popup.geometry("1200x800")

        # Create notebook for different visualizations
        notebook = ttk.Notebook(popup)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        if self.evaluation_results is not None and MATPLOTLIB_AVAILABLE:
            # Tab 1: Performance Comparison
            perf_frame = ttk.Frame(notebook)
            notebook.add(perf_frame, text="Performance Comparison")
            self._create_performance_viz(perf_frame)

            # Tab 2: Resource Efficiency
            resource_frame = ttk.Frame(notebook)
            notebook.add(resource_frame, text="Resource Efficiency")
            self._create_resource_viz(resource_frame)

            # Tab 3: Explainability
            explain_frame = ttk.Frame(notebook)
            notebook.add(explain_frame, text="Explainability")
            self._create_explainability_viz(explain_frame)

            # Tab 4: Combined Radar
            radar_frame = ttk.Frame(notebook)
            notebook.add(radar_frame, text="3D Radar Chart")
            self._create_radar_chart(radar_frame)

    def _create_algorithm_popup(self, model_name: str):
        """Create popup window for specific algorithm with detailed visualizations"""
        logger.info(f"[STEP] Creating popup for algorithm: {model_name}")
        logger.info(f"[INPUT] Model name: {model_name}")

        popup = tk.Toplevel(self.root)
        popup.title(f"Algorithm Details: {model_name}")
        popup.geometry("1400x900")

        if self.evaluation_results is None:
            messagebox.showwarning("No Data", "Evaluation results not available")
            return

        # Get model data
        model_data = self.evaluation_results[self.evaluation_results['model_name'] == model_name]
        if model_data.empty:
            logger.warning(f"[WARNING] No data found for model: {model_name}")
            messagebox.showwarning("Not Found", f"No data found for {model_name}")
            return

        row = model_data.iloc[0]
        logger.info(f"[OUTPUT] Model data retrieved: {len(model_data)} row(s)")

        # Create notebook
        notebook = ttk.Notebook(popup)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: Metrics Overview
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text="Metrics")
        self._create_metrics_tab(metrics_frame, row, model_name)

        # Tab 2: Visualizations
        if MATPLOTLIB_AVAILABLE:
            viz_frame = ttk.Frame(notebook)
            notebook.add(viz_frame, text="Visualizations")
            self._create_algorithm_viz_tab(viz_frame, row, model_name)

        # Tab 3: Input/Output Analysis
        io_frame = ttk.Frame(notebook)
        notebook.add(io_frame, text="Input/Output Analysis")
        self._create_io_analysis_tab(io_frame, model_name)

    def _create_metrics_tab(self, frame, row, model_name: str):
        """Create metrics overview tab"""
        text = scrolledtext.ScrolledText(frame, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        content = f"=== {model_name.upper()} - DETAILED METRICS ===\n\n"

        # Dimension 1: Performance
        content += "📊 DIMENSION 1: DETECTION PERFORMANCE\n"
        content += "=" * 60 + "\n"
        if 'f1_score' in row:
            content += f"F1 Score: {row['f1_score']:.6f}\n"
        if 'precision' in row:
            content += f"Precision (Pr): {row['precision']:.6f}\n"
        if 'recall' in row:
            content += f"Recall (Rc): {row['recall']:.6f}\n"
        if 'accuracy' in row:
            content += f"Accuracy: {row['accuracy']:.6f}\n"
        content += "\n"

        # Dimension 2: Resources
        content += "⚡ DIMENSION 2: RESOURCE EFFICIENCY\n"
        content += "=" * 60 + "\n"
        if 'training_time_seconds' in row:
            content += f"Training Time: {row['training_time_seconds']:.4f} seconds\n"
        if 'memory_used_mb' in row:
            content += f"Memory Used: {row['memory_used_mb']:.4f} MB\n"
        if 'peak_memory_mb' in row:
            content += f"Peak Memory: {row['peak_memory_mb']:.4f} MB\n"
        content += "\n"

        # Dimension 3: Explainability
        content += "🔍 DIMENSION 3: EXPLAINABILITY\n"
        content += "=" * 60 + "\n"
        if 'explainability_score' in row:
            content += f"Explainability Score: {row['explainability_score']:.6f}\n"
        if 'shap_score' in row and pd.notna(row['shap_score']):
            content += f"SHAP Score: {row['shap_score']:.6f}\n"
        if 'lime_score' in row and pd.notna(row['lime_score']):
            content += f"LIME Score: {row['lime_score']:.6f}\n"
        if 'native_interpretability' in row:
            content += f"Native Interpretability: {row['native_interpretability']:.1f}\n"
        content += "\n"

        text.insert(tk.END, content)
        text.config(state=tk.DISABLED)

    def _create_algorithm_viz_tab(self, frame, row, model_name: str):
        """Create visualization tab for algorithm"""
        if not MATPLOTLIB_AVAILABLE:
            return

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{model_name} - Performance Metrics', fontsize=14, fontweight='bold')

        # 1. Performance metrics bar chart
        ax1 = axes[0, 0]
        metrics = ['F1 Score', 'Precision', 'Recall', 'Accuracy']
        values = []
        for m in ['f1_score', 'precision', 'recall', 'accuracy']:
            if m in row:
                values.append(row[m])
            else:
                values.append(0)

        bars = ax1.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_ylim(0, max(1.0, max(values) * 1.1))
        ax1.set_title('Detection Performance Metrics')
        ax1.set_ylabel('Score')
        ax1.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom')

        # 2. Resource efficiency
        ax2 = axes[0, 1]
        resource_metrics = ['Time (s)', 'Memory (MB)']
        resource_values = []
        if 'training_time_seconds' in row:
            resource_values.append(row['training_time_seconds'])
        else:
            resource_values.append(0)
        if 'memory_used_mb' in row:
            resource_values.append(row['memory_used_mb'])
        else:
            resource_values.append(0)

        ax2.bar(resource_metrics, resource_values, color=['#9467bd', '#8c564b'])
        ax2.set_title('Resource Usage')
        ax2.set_ylabel('Value')
        ax2.grid(axis='y', alpha=0.3)

        # 3. Explainability breakdown
        ax3 = axes[1, 0]
        exp_components = []
        exp_values = []
        if 'native_interpretability' in row:
            exp_components.append('Native')
            exp_values.append(row['native_interpretability'])
        if 'shap_score' in row and pd.notna(row['shap_score']):
            exp_components.append('SHAP')
            exp_values.append(row['shap_score'])
        if 'lime_score' in row and pd.notna(row['lime_score']):
            exp_components.append('LIME')
            exp_values.append(row['lime_score'])
        if 'explainability_score' in row:
            exp_components.append('Combined')
            exp_values.append(row['explainability_score'])

        if exp_components:
            cmap = plt.get_cmap('viridis')
            ax3.bar(exp_components, exp_values, color=cmap(np.linspace(0, 1, len(exp_components))))
            ax3.set_title('Explainability Components')
            ax3.set_ylabel('Score')
            ax3.set_ylim(0, 1.1)
            ax3.grid(axis='y', alpha=0.3)

        # 4. 3D Score visualization
        ax4 = axes[1, 1]
        dimensions = ['Performance', 'Efficiency', 'Explainability']

        # Normalize scores for visualization
        perf_norm = row.get('f1_score', 0)
        eff_norm = 1 - (row.get('training_time_seconds', 100) / 100) if row.get('training_time_seconds', 0) > 0 else 0
        eff_norm = max(0, min(1, eff_norm))  # Clamp to [0, 1]
        exp_norm = row.get('explainability_score', 0)

        scores = [perf_norm, eff_norm, exp_norm]
        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
        angles += angles[:1]
        scores += scores[:1]

        ax4.plot(angles, scores, 'o-', linewidth=2, color='#e377c2')
        ax4.fill(angles, scores, alpha=0.25, color='#e377c2')
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(dimensions)
        ax4.set_ylim(0, 1)
        ax4.set_title('3D Score Overview')
        ax4.grid(True)

        plt.tight_layout()

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_io_analysis_tab(self, frame, model_name: str):
        """Create input/output analysis tab"""
        text = scrolledtext.ScrolledText(frame, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        content = f"=== {model_name.upper()} - INPUT/OUTPUT ANALYSIS ===\n\n"

        # Try to load preprocessing info
        if self.preprocessing_stats:
            content += "📥 INPUT DATA CHARACTERISTICS\n"
            content += "=" * 60 + "\n"
            content += f"Total Input Samples: {self.preprocessing_stats.get('total_samples', 'N/A')}\n"
            content += f"Total Input Features: {self.preprocessing_stats.get('total_features', 'N/A')}\n"
            if 'class_distribution' in self.preprocessing_stats:
                content += "\nInput Class Distribution:\n"
                for label, count in self.preprocessing_stats['class_distribution'].items():
                    total = self.preprocessing_stats['total_samples']
                    pct = (count / total * 100) if total > 0 else 0
                    content += f"  Class {label}: {count:,} ({pct:.2f}%)\n"
            content += "\n"

        # Output characteristics
        if self.evaluation_results is not None:
            model_row = self.evaluation_results[self.evaluation_results['model_name'] == model_name]
            if not model_row.empty:
                row = model_row.iloc[0]
                content += "📤 OUTPUT CHARACTERISTICS\n"
                content += "=" * 60 + "\n"
                content += f"Prediction Accuracy: {row.get('accuracy', 'N/A')}\n"
                content += f"F1 Score (Detection Rate): {row.get('f1_score', 'N/A')}\n"
                content += f"Precision (False Positive Rate): {row.get('precision', 'N/A')}\n"
                content += f"Recall (True Positive Rate): {row.get('recall', 'N/A')}\n"
                content += "\n"

        content += "💡 INTERPRETATION\n"
        content += "=" * 60 + "\n"
        content += "Input: Preprocessed dataset with harmonized features from TON_IoT and CIC-DDoS2019\n"
        content += "Output: Binary classification (Normal/Attack) with probability scores\n"
        content += "Processing: SMOTE balancing + RobustScaler normalization applied\n"

        text.insert(tk.END, content)
        text.config(state=tk.DISABLED)

    def _create_performance_viz(self, frame):
        """Create performance comparison visualization"""
        if not MATPLOTLIB_AVAILABLE or self.evaluation_results is None:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        models = self.evaluation_results['model_name'].values
        metrics = ['f1_score', 'precision', 'recall', 'accuracy']
        x = np.arange(len(models))
        width = 0.2

        for i, metric in enumerate(metrics):
            if metric in self.evaluation_results.columns:
                values = self.evaluation_results[metric].values
                ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())

        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_resource_viz(self, frame):
        """Create resource efficiency visualization"""
        if not MATPLOTLIB_AVAILABLE or self.evaluation_results is None:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        models = self.evaluation_results['model_name'].values

        if 'training_time_seconds' in self.evaluation_results.columns:
            ax1.bar(models, self.evaluation_results['training_time_seconds'], color='#d62728')
            ax1.set_ylabel('Time (seconds)')
            ax1.set_title('Training Time')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis='y', alpha=0.3)

        if 'memory_used_mb' in self.evaluation_results.columns:
            ax2.bar(models, self.evaluation_results['memory_used_mb'], color='#9467bd')
            ax2.set_ylabel('Memory (MB)')
            ax2.set_title('Memory Usage')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_explainability_viz(self, frame):
        """Create explainability visualization"""
        if not MATPLOTLIB_AVAILABLE or self.evaluation_results is None:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        models = self.evaluation_results['model_name'].values
        x = np.arange(len(models))
        width = 0.25

        if 'native_interpretability' in self.evaluation_results.columns:
            ax.bar(x - width, self.evaluation_results['native_interpretability'], width,
                  label='Native', color='#8c564b')

        if 'shap_score' in self.evaluation_results.columns:
            shap_vals = self.evaluation_results['shap_score'].fillna(0)
            ax.bar(x, shap_vals, width, label='SHAP', color='#e377c2')

        if 'explainability_score' in self.evaluation_results.columns:
            ax.bar(x + width, self.evaluation_results['explainability_score'], width,
                  label='Combined', color='#7f7f7f')

        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Score')
        ax.set_title('Explainability Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_radar_chart(self, frame):
        """Create combined 3D radar chart"""
        if not MATPLOTLIB_AVAILABLE or self.evaluation_results is None:
            return

        # Try to load dimension scores
        dim_scores_path = self.output_dir / 'phase3_evaluation' / 'dimension_scores.csv'
        if dim_scores_path.exists():
            dim_scores = pd.read_csv(dim_scores_path)
        else:
            # Calculate from evaluation results
            dim_scores = self.evaluation_results.copy()
            dim_scores['detection_performance'] = dim_scores['f1_score'] if 'f1_score' in dim_scores.columns else 0
            if 'training_time_seconds' in dim_scores.columns:
                max_time = dim_scores['training_time_seconds'].max()
                if pd.isna(max_time) or max_time <= 0:
                    dim_scores['resource_efficiency'] = 0.0
                else:
                    dim_scores['resource_efficiency'] = 1 - (dim_scores['training_time_seconds'] / max_time)
                dim_scores['resource_efficiency'] = dim_scores['resource_efficiency'].clip(lower=0, upper=1)
            else:
                dim_scores['resource_efficiency'] = 0.0
            dim_scores['explainability'] = dim_scores['explainability_score'] if 'explainability_score' in dim_scores.columns else 0

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

        categories = ['Detection\nPerformance', 'Resource\nEfficiency', 'Explainability']
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(dim_scores)))

        for idx, row in dim_scores.iterrows():
            model_name = row.get('model_name', f'Model {idx}')
            values = [
                row.get('detection_performance', 0),
                row.get('resource_efficiency', 0),
                row.get('explainability', 0)
            ]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title('3D Evaluation Framework - Combined View', pad=20, fontweight='bold')

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run(self):
        """Run the GUI"""
        logger.info("[ACTION] Starting Tkinter GUI main loop")
        self.root.mainloop()


def main():
    """Main function to launch the visualizer"""
    import argparse

    parser = argparse.ArgumentParser(description="IRP Results Visualizer")
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory containing results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        print(f"Error: Output directory {output_dir} does not exist")
        return

    app = ResultsVisualizer(output_dir)
    app.run()


if __name__ == "__main__":
    main()
