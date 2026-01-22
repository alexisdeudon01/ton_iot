import tkinter as tk
from tkinter import ttk
from gui.base_canvas import BaseCanvas

class PipelineCanvas(BaseCanvas):
    """
    Onglet affichant l'avancement des étapes du pipeline.
    """
    def setup_ui(self):
        self.label = ttk.Label(self.frame, text="État du Pipeline", font=("Helvetica", 14, "bold"))
        self.label.pack(pady=10)
        
        self.status_list = tk.Listbox(self.frame, height=15, width=70, font=("Helvetica", 10))
        self.status_list.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    def update_data(self, step_info: str):
        self.status_list.insert(tk.END, f"» {step_info}")
        self.status_list.see(tk.END)
