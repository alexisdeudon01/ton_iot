import tkinter as tk
from tkinter import ttk
from src.app.ui.panels.base_panel import AbstractPanel
from src.app.ui.state import UIState

class LogsPanel(AbstractPanel):
    def setup_ui(self):
        self.text = tk.Text(self, wrap="none", state="disabled", height=20)
        self.text.pack(side="left", fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        scrollbar.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=scrollbar.set)
        
        self.last_log_count = 0

    def update_state(self, state: UIState):
        if len(state.logs) > self.last_log_count:
            self.text.configure(state="normal")
            for i in range(self.last_log_count, len(state.logs)):
                log = state.logs[i]
                line = f"[{log.get('level', 'INFO')}] {log.get('action', '')}: {log.get('message', '')}\n"
                self.text.insert("end", line)
            self.text.configure(state="disabled")
            self.text.see("end")
            self.last_log_count = len(state.logs)
