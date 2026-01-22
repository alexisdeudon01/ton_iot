import tkinter as tk
from tkinter import ttk
from src.app.ui.panels.base_panel import AbstractPanel
from src.app.ui.state import UIState

class GraphPanel(AbstractPanel):
    def setup_ui(self):
        self.text = tk.Text(self, wrap="none", state="disabled")
        self.text.pack(fill="both", expand=True)
        self.last_mmd = ""

    def update_state(self, state: UIState):
        if state.pipeline_mmd != self.last_mmd:
            self.text.configure(state="normal")
            self.text.delete("1.0", "end")
            self.text.insert("1.0", state.pipeline_mmd)
            self.text.configure(state="disabled")
            self.last_mmd = state.pipeline_mmd
