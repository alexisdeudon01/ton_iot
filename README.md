# Llama 4 Maverick - DDoS Pipeline

Projet Python modulaire pour la détection de DDoS sur les datasets CICDDoS2019 et TON_IoT.

## Architecture
- **Clean Architecture**: Séparation stricte entre contrats, ports (interfaces) et infrastructure.
- **Task DAG**: Pipeline orchestré via un graphe de dépendances.
- **Event Bus**: Communication asynchrone entre le pipeline et l'UI.
- **Tkinter GUI**: Interface temps réel pour le monitoring.

## Algorithmes
- LR, DT, RF, CNN, TabNet.

## Utilisation
```bash
python -m src.app.ui.main
