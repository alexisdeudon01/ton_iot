"""
Central configuration for the New DDoS Detection Pipeline
Defines hyperparameters, XAI methods, and paths.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend for multi-threading safety

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RR_DIR = ROOT_DIR / "rr"
DATA_PATH = ROOT_DIR / "datasets/ton_iot/train_test_network.csv"

# Algorithms to use
ALGORITHMS = ['LR', 'DT', 'RF', 'CNN', 'TabNet']

# Hyperparameter Grids for Tuning (Phase 3)
HYPERPARAMS = {
    'LR': {
        'C': [0.1, 1.0, 10.0],
        'max_iter': [1000]
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
}

# XAI Configuration (Phase 4)
XAI_METHODS = ['SHAP', 'LIME', 'FI']

# XAI Validation Criteria Weights
XAI_CRITERIA_WEIGHTS = {
    'fidelity': 0.4,
    'stability': 0.4,
    'complexity': 0.2
}
