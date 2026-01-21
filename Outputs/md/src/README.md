# Source Modules (src/)

This directory contains all the core Python modules for the IRP Research Pipeline.

## Modules

- **`main_pipeline.py`** - Main IRP pipeline orchestrator (Phases 1, 3, 5)
- **`dataset_loader.py`** - Dataset loading (TON_IoT, CIC-DDoS2019)
- **`data_harmonization.py`** - Feature harmonization and early fusion
- **`preprocessing_pipeline.py`** - Data preprocessing (SMOTE, RobustScaler, CV)
- **`evaluation_3d.py`** - 3D evaluation framework (Performance, Resources, Explainability)
- **`ahp_topsis_framework.py`** - AHP-TOPSIS multi-criteria ranking
- **`models/`** - ML/DL models package
  - **`models/cnn.py`** - CNN model for tabular data
  - **`models/tabnet.py`** - TabNet model wrapper
  - **`models/registry.py`** - Model registry (single source of truth)
  - **`models/sklearn_models.py`** - Scikit-learn models (LR, DT, RF)

## Usage

All modules are imported by `main.py` at the project root. The `src/` directory is automatically added to Python path when running `main.py`.

## Structure

```
src/
├── __init__.py              # Package initialization
├── main_pipeline.py         # Main pipeline
├── dataset_loader.py        # Dataset loading
├── data_harmonization.py    # Harmonization
├── preprocessing_pipeline.py # Preprocessing
├── evaluation_3d.py         # 3D evaluation
├── ahp_topsis_framework.py  # AHP-TOPSIS
├── models/                  # Models package
│   ├── __init__.py
│   ├── cnn.py              # CNN model
│   ├── tabnet.py           # TabNet model
│   ├── registry.py         # Model registry
│   └── sklearn_models.py   # Scikit-learn models
```
