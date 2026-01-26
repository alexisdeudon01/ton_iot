from __future__ import annotations

from pathlib import Path

from src.visualization.data_viz import generate_data_figures
from src.visualization.model_viz import generate_model_figures
from src.visualization.mcdm_viz import generate_mcdm_figures


def generate_all_figures() -> Path:
    graphs_dir = Path("graphs")
    graphs_dir.mkdir(parents=True, exist_ok=True)

    generate_data_figures(graphs_dir)
    generate_model_figures(graphs_dir)
    generate_mcdm_figures(graphs_dir)

    return graphs_dir


__all__ = ["generate_all_figures"]
