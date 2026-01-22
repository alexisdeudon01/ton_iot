import tkinter as tk
from tkinter import ttk
from src.app.ui.controller import UIController
from src.app.ui.state import UIState
from src.app.ui.panels.logs_panel import LogsPanel
from src.app.ui.panels.config_panel import ConfigPanel
from src.app.ui.panels.graph_panel import GraphPanel
from src.app.ui.panels.dataset_panel import DatasetPanel
from src.app.ui.panels.dist_panel import DistPanel
from src.app.ui.panels.timeline_panel import TimelinePanel

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Llama 4 Maverick - DDoS Pipeline Monitor")
        self.geometry("1200x800")
        
        self.controller = UIController(self.update_state)
        self.setup_ui()
        
        # Poll for events
        self.after(50, self.poll_events)

    def setup_ui(self):
        # Toolbar
        self.toolbar = ttk.Frame(self)
        self.toolbar.pack(side="top", fill="x")
        
        self.start_btn = ttk.Button(self.toolbar, text="Start Pipeline", command=self.on_start)
        self.start_btn.pack(side="left", padx=5, pady=5)
        
        self.status_label = ttk.Label(self.toolbar, text="Ready")
        self.status_label.pack(side="right", padx=5)
        
        # Notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        
        self.panels = {
            "Logs": LogsPanel(self.notebook, self.controller),
            "Config": ConfigPanel(self.notebook, self.controller),
            "Graph": GraphPanel(self.notebook, self.controller),
            "Datasets": DatasetPanel(self.notebook, self.controller),
            "Distributions": DistPanel(self.notebook, self.controller),
            "Timeline": TimelinePanel(self.notebook, self.controller)
        }
        
        for name, panel in self.panels.items():
            self.notebook.add(panel, text=name)

    def on_start(self):
        self.controller.start_pipeline("configs/pipeline.yaml")
        self.start_btn.configure(state="disabled")

    def update_state(self, state: UIState):
        # This is called from the event bus thread, but Tkinter needs UI thread
        # However, since we use a queue and poll, we don't need to worry about this here
        # if we were calling it directly. But QueueEventBus calls it from its thread.
        # So we should use a thread-safe way to update UI.
        pass

    def poll_events(self):
        # In this implementation, the controller updates its internal state
        # and we just refresh the panels.
        state = self.controller.state
        self.status_label.configure(text=state.status_message)
        
        for panel in self.panels.values():
            panel.update_state(state)
            
        self.after(50, self.poll_events)
