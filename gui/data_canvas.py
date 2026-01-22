import tkinter as tk
from tkinter import ttk
from gui.base_canvas import BaseCanvas

class DataCanvas(BaseCanvas):
    """
    Onglet affichant les statistiques des datasets (shape, labels, NaN).
    """
    def setup_ui(self):
        self.label = ttk.Label(self.frame, text="Statistiques des Données", font=("Helvetica", 14, "bold"))
        self.label.pack(pady=10)
        
        self.tree = ttk.Treeview(self.frame, columns=("Metric", "Value"), show="headings")
        self.tree.heading("Metric", text="Métrique")
        self.tree.heading("Value", text="Valeur")
        self.tree.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    def update_data(self, stats: dict):
        # Nettoyage de l'affichage précédent
        for i in self.tree.get_children():
            self.tree.delete(i)
            
        # Insertion des nouvelles statistiques
        for k, v in stats.items():
            self.tree.insert("", tk.END, values=(k, v))
