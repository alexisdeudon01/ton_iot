import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.app.ui.panels.base_panel import AbstractPanel
from src.app.ui.state import UIState

class DistPanel(AbstractPanel):
    def setup_ui(self):
        self.top_frame = ttk.Frame(self)
        self.top_frame.pack(fill="x")
        
        ttk.Label(self.top_frame, text="Feature:").pack(side="left")
        self.feature_var = tk.StringVar()
        self.feature_combo = ttk.Combobox(self.top_frame, textvariable=self.feature_var)
        self.feature_combo.pack(side="left", padx=5)
        self.feature_combo.bind("<<ComboboxSelected>>", self.on_feature_change)
        
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.current_artifact = "cic_projected" # Default
        self.known_features = []

    def on_feature_change(self, event):
        feature = self.feature_var.get()
        self.controller.request_distribution(self.current_artifact, feature)

    def update_state(self, state: UIState):
        # Update feature list if needed
        if state.config and not self.known_features:
            # Try to get features from alignment or first artifact
            if state.artifacts:
                self.known_features = state.artifacts[0].columns
                self.feature_combo["values"] = self.known_features
        
        # Check if new distribution is ready
        feature = self.feature_var.get()
        key = f"{self.current_artifact}:{feature}"
        if key in state.last_distributions:
            bundle = state.last_distributions[key]
            self.ax.clear()
            self.ax.bar(bundle.bins[:-1], bundle.counts, width=(bundle.bins[1]-bundle.bins[0]))
            self.ax.set_title(f"Distribution of {feature}")
            self.canvas.draw()
