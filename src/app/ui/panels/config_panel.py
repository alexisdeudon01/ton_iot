import tkinter as tk
from tkinter import ttk
import yaml
from src.app.ui.panels.base_panel import AbstractPanel
from src.app.ui.state import UIState

class ConfigPanel(AbstractPanel):
    def setup_ui(self):
        self.text = tk.Text(self, wrap="none", state="disabled")
        self.text.pack(fill="both", expand=True)
        self.has_config = False

    def update_state(self, state: UIState):
        if state.config and not self.has_config:
            self.text.configure(state="normal")
            self.text.delete("1.0", "end")
            config_yaml = yaml.dump(state.config.model_dump(), default_flow_style=False)
            self.text.insert("1.0", config_yaml)
            self.text.configure(state="disabled")
            self.has_config = True
