# DDoS Pipeline (Clean Architecture + Late Fusion)

Projet Python modulaire pour la détection de DDoS sur les datasets CICDDoS2019 et TON_IoT.

## Architecture
- **Core**: Contrats (Pydantic), Moteur de DAG, Interfaces (Ports).
- **Infra**: Adaptateurs Polars, Sklearn, PyTorch, TabNet, EventBus.
- **App**: Tâches de la pipeline et Interface GUI Tkinter.

## Algorithmes supportés
- Logistic Regression (LR)
- Decision Tree (DT)
- Random Forest (RF)
- CNN (PyTorch)
- TabNet (pytorch-tabnet)

## Installation
```bash
pip install polars pydantic pyyaml psutil scikit-learn torch pytorch-tabnet joblib matplotlib
```

## Utilisation
### Pipeline CLI
```bash
python -m src.app.pipeline.main --config configs/pipeline.yaml
```

### Pipeline avec GUI Monitor
```bash
python -m src.app.pipeline.main --config configs/pipeline.yaml --ui
```

## Tests
```bash
pytest tests/
