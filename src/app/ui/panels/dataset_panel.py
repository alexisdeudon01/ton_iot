import tkinter as tk
from tkinter import ttk
from src.app.ui.panels.base_panel import AbstractPanel
from src.app.ui.state import UIState

class DatasetPanel(AbstractPanel):
    def setup_ui(self):
        self.tree = ttk.Treeview(self, columns=("artifact", "rows", "cols", "dataset"), show="headings")
        self.tree.heading("artifact", text="Artifact ID")
        self.tree.heading("rows", text="Rows")
        self.tree.heading("cols", text="Cols")
        self.tree.heading("dataset", text="Dataset")
        self.tree.pack(fill="both", expand=True)
        self.known_artifacts = set()

    def update_state(self, state: UIState):
        for art in state.artifacts:
            if art.artifact_id not in self.known_artifacts:
                dataset = "N/A"
                if "cic" in art.artifact_id: dataset = "CIC"
                elif "ton" in art.artifact_id: dataset = "TON"
                
                self.tree.insert("", "end", values=(art.artifact_id, art.n_rows, art.n_cols, dataset))
                self.known_artifacts.add(art.artifact_id)
