# Source Modules (src/)

This directory contains all the core Python modules for the IRP Research Pipeline.

## Modules

- **`main_pipeline.py`** - Main IRP pipeline orchestrator (Phases 1, 3, 5)
- **`dataset_loader.py`** - Dataset loading (TON_IoT, CIC-DDoS2019)
- **`data_harmonization.py`** - Feature harmonization and early fusion
- **`preprocessing_pipeline.py`** - Data preprocessing (SMOTE, RobustScaler, CV)
- **`evaluation_3d.py`** - 3D evaluation framework (Performance, Resources, Explainability)
- **`ahp_topsis_framework.py`** - AHP-TOPSIS multi-criteria ranking
- **`models_cnn.py`** - CNN model for tabular data
- **`models_tabnet.py`** - TabNet model wrapper

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
├── models_cnn.py           # CNN model
└── models_tabnet.py        # TabNet model
```
