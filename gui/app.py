import tkinter as tk
from tkinter import ttk
import threading
import time
from gui.logs_canvas import LogsCanvas
from gui.data_canvas import DataCanvas
from gui.pipeline_canvas import PipelineCanvas
from gui.xai_canvas import XAICanvas

class DDoSApp:
    """
    Application Tkinter principale pour la visualisation temps réel du pipeline.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Llama 4 Maverick - DDoS Detection Pipeline")
        self.root.geometry("1000x750")
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Initialisation des onglets
        self.logs_tab = LogsCanvas(self.notebook, "Logs")
        self.data_tab = DataCanvas(self.notebook, "Données")
        self.pipeline_tab = PipelineCanvas(self.notebook, "Pipeline")
        self.xai_tab = XAICanvas(self.notebook, "XAI")
        
        # Barre de statut pour les ressources
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.resource_label = ttk.Label(self.status_frame, text="CPU: 0% | RAM: 0 MB", font=("Courier", 10))
        self.resource_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.progress = ttk.Progressbar(self.status_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack(side=tk.RIGHT, padx=10, pady=5)

    def update_resources(self, cpu: float, ram: float):
        """Met à jour l'affichage des ressources en bas de fenêtre."""
        self.resource_label.config(text=f"CPU: {cpu}% | RAM: {ram:.2f} MB")

    def log(self, message: str):
        """Ajoute une ligne dans l'onglet Logs."""
        self.logs_tab.update_data(message)

    def update_pipeline(self, step: str, progress_val: int = None):
        """Met à jour l'état d'avancement du pipeline."""
        self.pipeline_tab.update_data(step)
        if progress_val is not None:
            self.progress['value'] = progress_val

    def update_data_stats(self, stats: dict):
        """Met à jour le tableau des statistiques de données."""
        self.data_tab.update_data(stats)

    def update_xai(self, shap_data: dict):
        """Met à jour le graphique SHAP."""
        self.xai_tab.update_data(shap_data)
