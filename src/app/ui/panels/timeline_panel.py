import tkinter as tk
from tkinter import ttk
from src.app.ui.panels.base_panel import AbstractPanel
from src.app.ui.state import UIState

class TimelinePanel(AbstractPanel):
    def setup_ui(self):
        self.tree = ttk.Treeview(self, columns=("task", "status", "duration"), show="headings")
        self.tree.heading("task", text="Task Name")
        self.tree.heading("status", text="Status")
        self.tree.heading("duration", text="Duration (s)")
        self.tree.pack(fill="both", expand=True)
        self.task_items = {} # task_name -> item_id

    def update_state(self, state: UIState):
        for task_name, status in state.task_status.items():
            duration_str = f"{status.duration:.2f}" if status.duration > 0 else "-"
            values = (task_name, status.status, duration_str)
            
            if task_name in self.task_items:
                self.tree.item(self.task_items[task_name], values=values)
            else:
                item_id = self.tree.insert("", "end", values=values)
                self.task_items[task_name] = item_id
