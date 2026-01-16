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
        
        # Matrix details button
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=5)
        
        btn_details = tk.Button(btn_frame, text="Show Detailed Matrices", 
                               command=self._show_matrix_details)
        btn_details.pack(side=tk.LEFT, padx=5)
    
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
        
        if self.evaluation_results is not None:
            details += "📊 DETECTION PERFORMANCE MATRIX (Dimension 1)\n"
            details += "=" * 60 + "\n\n"
            
            # Performance metrics matrix
            perf_cols = ['model_name', 'f1_score', 'accuracy', 'precision', 'recall']
            available_perf = [col for col in perf_cols if col in self.evaluation_results.columns]
            
            if available_perf:
                perf_df = self.evaluation_results[available_perf].copy()
                details += "Performance Metrics Table:\n"
                details += perf_df.to_string(index=False)
                details += "\n\n"
                
                # Confusion Matrix Details for each model
                details += "🔍 CONFUSION MATRICES DETAILS:\n"
                details += "-" * 60 + "\n"
                
                # Try to load detailed results from algorithm reports or JSON if available
                for idx, row in perf_df.iterrows():
                    model_name = row['model_name']
                    details += f"\n{model_name}:\n"
                    details += f"  F1 Score: {row['f1_score']:.4f}\n"
                    details += f"  Precision: {row['precision']:.4f}\n"
                    details += f"  Recall: {row['recall']:.4f}\n"
                    details += f"  Accuracy: {row['accuracy']:.4f}\n"
                
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
    
    def run(self):
        """Run the GUI"""
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
