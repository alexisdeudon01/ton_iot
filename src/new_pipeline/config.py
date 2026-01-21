"""
Configuration centralisée pour le nouveau pipeline de détection DDoS.
Utilise Pydantic pour la validation des données et des chemins.
"""
from pathlib import Path
from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field, validator
import matplotlib

matplotlib.use('Agg')  # Force le backend non-interactif

class PipelineConfig(BaseModel):
    """Modèle de configuration validé par Pydantic"""
    
    # Chemins
    root_dir: Path = Field(default=Path(__file__).resolve().parent.parent.parent)
    rr_dir: Path = Field(default=Path(__file__).resolve().parent.parent.parent / "rr")
    ton_iot_path: Path = Field(default=Path(__file__).resolve().parent.parent.parent / "datasets/ton_iot/train_test_network.csv")
    cic_ddos_dir: Path = Field(default=Path(__file__).resolve().parent.parent.parent / "datasets/cic_ddos2019")
    parquet_dir: Path = Field(default=Path(__file__).resolve().parent.parent.parent / "datasets/parquet")
    models_dir: Path = Field(default=Path(__file__).resolve().parent.parent.parent / "outputs/models")

    # Algorithmes
    algorithms: List[Literal['LR', 'DT', 'RF', 'KNN', 'CNN', 'TabNet']] = [
        'LR', 'DT', 'RF', 'KNN', 'CNN', 'TabNet'
    ]

    # Hyperparamètres
    hyperparams: Dict[str, Dict[str, List[Any]]] = Field(default_factory=lambda: {
        'LR': {
            'C': [0.1, 1.0, 10.0],
            'max_iter': [1000]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        },
        'DT': {
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10]
        },
        'RF': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None]
        },
        'TabNet': {
            'n_d': [8, 16, 32],
            'n_a': [8, 16, 32]
        },
        'CNN': {
            'lr': [0.001, 0.01],
            'epochs': [10, 20]
        }
    })

    # XAI
    xai_methods: List[str] = ['SHAP', 'LIME', 'FI']
    xai_criteria_weights: Dict[str, float] = {
        'fidelity': 0.4,
        'stability': 0.4,
        'complexity': 0.2
    }

    # Ressources
    max_memory_percent: float = Field(50.0, ge=10.0, le=90.0)
    dask_workers: int = Field(2, ge=1, le=8)
    random_state: int = 42

    @validator('ton_iot_path', 'cic_ddos_dir')
    def validate_paths_exist(cls, v):
        if not v.exists():
            # On ne lève pas d'erreur ici car le data_loader peut gérer la conversion
            # mais on prévient
            import warnings
            warnings.warn(f"Chemin source non trouvé: {v}")
        return v

    class Config:
        validate_assignment = True

# Instance globale pour compatibilité descendante (si nécessaire) ou usage direct
config = PipelineConfig()

# Export des constantes pour compatibilité avec le code existant
ROOT_DIR = config.root_dir
RR_DIR = config.rr_dir
TON_IOT_PATH = config.ton_iot_path
CIC_DDOS_DIR = config.cic_ddos_dir
ALGORITHMS = config.algorithms
HYPERPARAMS = config.hyperparams
XAI_METHODS = config.xai_methods
XAI_CRITERIA_WEIGHTS = config.xai_criteria_weights
