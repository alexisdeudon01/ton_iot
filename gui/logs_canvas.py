import tkinter as tk
from tkinter import ttk
from gui.base_canvas import BaseCanvas

class LogsCanvas(BaseCanvas):
    """
    Onglet affichant les logs structurés du pipeline en temps réel.
    """
    def setup_ui(self):
        self.text_area = tk.Text(self.frame, height=20, width=80, state='disabled', font=("Courier", 9))
        self.text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.scrollbar = ttk.Scrollbar(self.frame, command=self.text_area.yview)
        self.text_area.configure(yscrollcommand=self.scrollbar.set)

    def update_data(self, log_message: str):
        self.text_area.configure(state='normal')
        self.text_area.insert(tk.END, log_message + "\n")
        self.text_area.see(tk.END)
        self.text_area.configure(state='disabled')
