import tkinter as tk
from tkinter import ttk
from abc import ABC, abstractmethod

class BaseCanvas(ABC):
    """
    Classe de base abstraite pour les onglets de l'interface graphique.
    """
    def __init__(self, parent: ttk.Notebook, title: str):
        self.frame = ttk.Frame(parent)
        parent.add(self.frame, text=title)
        self.setup_ui()

    @abstractmethod
    def setup_ui(self):
        """Initialise les widgets de l'onglet."""
        pass

    @abstractmethod
    def update_data(self, data: any):
        """Met à jour l'affichage avec de nouvelles données."""
        pass
