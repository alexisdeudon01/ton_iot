import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from typing import Dict, Callable

class MCDMDashboard(tk.Frame):
    """
    Dashboard Tkinter pour visualiser le Front de Pareto et les scores TOPSIS.
    """
    def __init__(self, parent, config: Dict, run_callback: Callable):
        super().__init__(parent)
        self.config = config
        self.run_callback = run_callback
        self._setup_ui()

    def _setup_ui(self):
        # Panneau de contrôle (Gauche)
        control_frame = ttk.LabelFrame(self, text="Paramètres PME", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(control_frame, text="Volume de données (%)").pack(anchor=tk.W)
        self.data_slider = ttk.Scale(control_frame, from_=1, to=100, orient=tk.HORIZONTAL)
        self.data_slider.set(10)
        self.data_slider.pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Échantillonnage (lignes)").pack(anchor=tk.W)
        self.sample_entry = ttk.Entry(control_frame)
        self.sample_entry.insert(0, "5000")
        self.sample_entry.pack(fill=tk.X, pady=5)

        self.run_btn = ttk.Button(control_frame, text="Lancer l'Analyse", command=self._on_run)
        self.run_btn.pack(fill=tk.X, pady=20)

        # Zone d'affichage des graphiques (Droite)
        self.viz_frame = ttk.Frame(self)
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig, (self.ax_pareto, self.ax_topsis) = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _on_run(self):
        try:
            frac = self.data_slider.get() / 100.0
            n_samples = int(self.sample_entry.get())
            self.run_callback(frac, n_samples)
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

    def update_plots(self, all_results: pd.DataFrame, pareto_results: pd.DataFrame, topsis_results: pd.DataFrame):
        self.ax_pareto.clear()
        self.ax_topsis.clear()

        # 1. Graphique Pareto [Précision vs Ressources]
        # On utilise CPU comme proxy pour les ressources
        self.ax_pareto.scatter(all_results['cpu_percent'], all_results['f1'], color='gray', alpha=0.5, label='Modèles dominés')
        self.ax_pareto.scatter(pareto_results['cpu_percent'], pareto_results['f1'], color='blue', s=100, label='Front de Pareto')
        
        # Zone Optimale PME
        thresholds = self.config['thresholds_pme']
        self.ax_pareto.axvspan(0, thresholds['max_cpu_percent'], color='green', alpha=0.1, label='Zone Optimale PME')
        self.ax_pareto.axhline(y=thresholds['min_f1_score'], color='red', linestyle='--', label='Min F1 PME')

        self.ax_pareto.set_xlabel("Usage CPU (%)")
        self.ax_pareto.set_ylabel("F1-Score")
        self.ax_pareto.set_title("Frontière de Pareto (Efficience)")
        self.ax_pareto.legend(fontsize='small')

        # 2. Bar chart TOPSIS
        if not topsis_results.empty:
            models = topsis_results['model_name']
            scores = topsis_results['topsis_score']
            bars = self.ax_topsis.bar(models, scores, color='skyblue')
            self.ax_topsis.set_title("Classement Final TOPSIS")
            self.ax_topsis.set_ylabel("Score de Proximité Relative (Ci)")
            plt.setp(self.ax_topsis.get_xticklabels(), rotation=45, ha="right")
            
            # Highlight best
            bars[0].set_color('gold')

        self.fig.tight_layout()
        self.canvas.draw()
