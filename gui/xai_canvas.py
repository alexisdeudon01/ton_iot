import tkinter as tk
from tkinter import ttk
from gui.base_canvas import BaseCanvas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class XAICanvas(BaseCanvas):
    """
    Onglet affichant les visualisations d'explicabilité (SHAP).
    """
    def setup_ui(self):
        self.label = ttk.Label(self.frame, text="Explicabilité SHAP (Top-K)", font=("Helvetica", 14, "bold"))
        self.label.pack(pady=10)
        
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_data(self, shap_data: dict):
        """
        shap_data: {'features': list, 'importances': list}
        """
        self.ax.clear()
        features = shap_data.get('features', [])
        importances = shap_data.get('importances', [])
        
        if not features: return

        y_pos = np.arange(len(features))
        self.ax.barh(y_pos, importances, align='center', color='skyblue')
        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels(features)
        self.ax.invert_yaxis()
        self.ax.set_xlabel('Importance SHAP moyenne')
        self.ax.set_title('Top Caractéristiques')
        
        self.fig.tight_layout()
        self.canvas.draw()
