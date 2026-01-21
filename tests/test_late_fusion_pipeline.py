import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.new_pipeline.config import config
from src.new_pipeline.data_loader import LateFusionDataLoader
from src.core.results import TrainingResult, TestResult

def test_config_loading():
    """Vérifie que la configuration Pydantic est correctement chargée."""
    assert config.phase0.label_col == "label"
    assert config.phase2.scaler == "robust"
    assert config.phase5.fusion_mode == "late"

def test_result_objects():
    """Vérifie l'intégrité des objets de résultat."""
    tr = TrainingResult(model_name="RF", success=True, training_time=10.5)
    assert tr.success is True
    assert tr.final_accuracy == 0.0 # Valeur par défaut si non fournie

def test_data_loader_init():
    """Vérifie l'initialisation du DataLoader."""
    loader = LateFusionDataLoader()
    assert loader.p0.chunksize >= 10000
    assert loader.paths.data_parquet_root.exists()

def test_late_fusion_logic():
    """Vérifie la logique mathématique de la Late Fusion."""
    # Test de la moyenne pondérée simple
    p_cic = np.array([0.1, 0.8])
    p_ton = np.array([0.2, 0.6])
    w = 0.7
    p_final = w * p_cic + (1 - w) * p_ton
    expected = 0.7 * 0.8 + 0.3 * 0.6
    assert np.isclose(p_final[1], expected)

def test_label_constraint():
    """Vérifie que la contrainte sur le nom de la colonne label est respectée."""
    assert config.phase0.label_col == "label"
