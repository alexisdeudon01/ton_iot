import tkinter as tk
from tkinter import ttk
from abc import ABC, abstractmethod
from src.app.ui.state import UIState

class AbstractPanel(ttk.Frame, ABC):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.setup_ui()

    @abstractmethod
    def setup_ui(self):
        pass

    @abstractmethod
    def update_state(self, state: UIState):
        pass
