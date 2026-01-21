import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
import time
from typing import Optional
from src.app.ui.state import AppState
from src.app.ui.reducer import reduce_event
from src.core.events.models import PipelineEvent

class MainWindow:
    def __init__(self, root: tk.Tk, event_queue: queue.Queue):
        self.root = root
        self.event_queue = event_queue
        self.state = AppState()
        
        self.root.title("DDoS Pipeline Monitor")
        self.root.geometry("1200x800")
        
        self.setup_ui()
        self.poll_events()

    def setup_ui(self):
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Notebook for panels
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both")
        
        # Logs Panel
        self.logs_frame = ttk.Frame(self.notebook)
        self.logs_text = scrolledtext.ScrolledText(self.logs_frame, state='disabled', height=20)
        self.logs_text.pack(expand=True, fill="both")
        self.notebook.add(self.logs_frame, text="Logs")
        
        # Config Panel
        self.config_frame = ttk.Frame(self.notebook)
        self.config_text = scrolledtext.ScrolledText(self.config_frame, height=20)
        self.config_text.pack(expand=True, fill="both")
        self.notebook.add(self.config_frame, text="Config")
        
        # Timeline Panel
        self.timeline_frame = ttk.Frame(self.notebook)
        self.timeline_tree = ttk.Treeview(self.timeline_frame, columns=("Status", "Duration"), show="headings")
        self.timeline_tree.heading("Status", text="Status")
        self.timeline_tree.heading("Duration", text="Duration (s)")
        self.timeline_tree.pack(expand=True, fill="both")
        self.notebook.add(self.timeline_frame, text="Timeline")

    def poll_events(self):
        try:
            while True:
                event = self.event_queue.get_nowait()
                self.state = reduce_event(self.state, event)
                self.update_ui()
        except queue.Empty:
            pass
        self.root.after(50, self.poll_events)

    def update_ui(self):
        self.status_var.set(self.state.status_message)
        
        # Update logs
        self.logs_text.config(state='normal')
        self.logs_text.delete('1.0', tk.END)
        self.logs_text.insert(tk.END, "\n".join(self.state.logs))
        self.logs_text.see(tk.END)
        self.logs_text.config(state='disabled')
        
        # Update timeline
        for item in self.timeline_tree.get_children():
            self.timeline_tree.delete(item)
        for task, status in self.state.task_status.items():
            duration = self.state.task_durations.get(task, 0.0)
            self.timeline_tree.insert("", tk.END, text=task, values=(status, f"{duration:.2f}"))

def start_ui(event_queue: queue.Queue):
    root = tk.Tk()
    app = MainWindow(root, event_queue)
    root.mainloop()

if __name__ == "__main__":
    # For testing UI standalone
    q = queue.Queue()
    start_ui(q)
