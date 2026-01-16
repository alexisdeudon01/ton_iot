# Structure du Projet

## Vue d'ensemble

```
ton_iot/
├── main.py                          # Point d'entrée principal (CLI)
├── main_pipeline.py                 # Pipeline IRP principal
├── dataset_loader.py                # Chargement datasets (TON_IoT, CIC-DDoS2019)
├── data_harmonization.py            # Harmonisation et early fusion
├── preprocessing_pipeline.py        # Preprocessing (SMOTE, RobustScaler)
├── evaluation_3d.py                 # Évaluation 3D (Performance, Resources, Explainability)
├── ahp_topsis_framework.py          # AHP-TOPSIS ranking
├── models_cnn.py                    # Modèle CNN pour données tabulaires
├── models_tabnet.py                 # Modèle TabNet
│
├── README.md                        # Documentation principale
├── DIMENSIONS_CALCULATION.md        # Documentation calcul des 3 dimensions
├── OUTPUT_EXPECTED.md               # Documentation des outputs attendus
├── PROJECT_STRUCTURE.md             # Ce fichier
│
├── requirements.txt                 # Dépendances Python
├── ton_iot.code-workspace           # Configuration VS Code
│
├── datasets/                        # Datasets (nouvelle structure)
│   ├── ton_iot/                     # Dataset TON_IoT
│   │   ├── train_test_network.csv
│   │   └── windows10_dataset.csv    # (optionnel)
│   └── cic_ddos2019/                # Fichiers CSV CIC-DDoS2019
│       └── *.csv                    # Un fichier par type d'attaque
│
├── Processed_datasets/              # Datasets préprocessés (TON_IoT)
│   └── Processed_Windows_dataset/
│       └── windows10_dataset.csv
│
├── output/                          # Résultats générés
│   ├── phase1_preprocessing/        # Données préprocessées
│   ├── phase3_evaluation/           # Résultats évaluation 3D
│   │   ├── algorithm_reports/       # Rapports par algorithme
│   │   └── visualizations/          # Graphiques
│   ├── phase5_ranking/              # Résultats AHP-TOPSIS
│   └── logs/                        # Fichiers de logs
│
└── _old/                            # Fichiers archivés (non utilisés)
    ├── documents/                   # PDFs (papers, documentation)
    ├── legacy_scripts/              # Scripts Python obsolètes
    ├── web_artifacts/               # HTML/JS de téléchargements web
    ├── documentation_old/           # Ancienne documentation
    └── old_results/                 # Anciens résultats (results/)
```

## Fichiers Principaux

### Points d'entrée
- **`main.py`** : Point d'entrée principal avec CLI (`python main.py`)
- **`main_pipeline.py`** : Pipeline IRP complet (appelé par main.py)

### Modules Core
- **`dataset_loader.py`** : Charge TON_IoT et CIC-DDoS2019
- **`data_harmonization.py`** : Harmonisation et early fusion des datasets
- **`preprocessing_pipeline.py`** : Preprocessing (SMOTE, RobustScaler, CV)
- **`evaluation_3d.py`** : Évaluation 3D (génération rapports, visualisations)
- **`ahp_topsis_framework.py`** : AHP-TOPSIS pour ranking
- **`models_cnn.py`** : Implémentation CNN
- **`models_tabnet.py`** : Implémentation TabNet

### Documentation
- **`README.md`** : Documentation principale du projet
- **`DIMENSIONS_CALCULATION.md`** : Calculs détaillés des 3 dimensions
- **`OUTPUT_EXPECTED.md`** : Documentation des fichiers de sortie
- **`PROJECT_STRUCTURE.md`** : Ce fichier (structure du projet)

## Répertoires

### `datasets/`
Contient les datasets organisés :
- **`ton_iot/`** : Dataset TON_IoT
  - `train_test_network.csv` : Dataset principal (requis)
  - `windows10_dataset.csv` : Alternative/processed (optionnel)
- **`cic_ddos2019/`** : Fichiers CSV du dataset CIC-DDoS2019
  - Placez tous les fichiers CSV téléchargés ici
  - Le pipeline charge automatiquement tous les CSV et les combine

**Note** : Le pipeline vérifie aussi les emplacements legacy (`data/raw/`, racine) pour compatibilité.

### `output/`
Résultats générés par le pipeline :
- **`phase1_preprocessing/`** : Données préprocessées
- **`phase3_evaluation/`** : Résultats évaluation, rapports, visualisations
- **`phase5_ranking/`** : Résultats AHP-TOPSIS
- **`logs/`** : Fichiers de logs par phase

### `Processed_datasets/`
Datasets préprocessés (utilisés par certains scripts legacy)

### `_old/`
Fichiers archivés non utilisés :
- **`documents/`** : PDFs de référence
- **`legacy_scripts/`** : Scripts Python obsolètes
- **`web_artifacts/`** : Fichiers HTML/JS de téléchargements web
- **`documentation_old/`** : Ancienne documentation
- **`old_results/`** : Ancien répertoire `results/` (maintenant `output/`)

## Utilisation

### Lancer le pipeline complet
```bash
python main.py
```

### Lancer une phase spécifique
```bash
python main.py --phase 1  # Phase 1 seulement
python main.py --phase 3  # Phase 3 seulement
python main.py --phase 5  # Phase 5 seulement
```

### Structure de sortie
Tous les résultats sont générés dans `output/` :
- CSV de résultats
- Rapports Markdown par algorithme
- Visualisations PNG
- Logs détaillés

Voir `OUTPUT_EXPECTED.md` pour la documentation complète des outputs.

## Notes

- Le dossier `_old/` contient les fichiers archivés mais peut être ignoré pour le développement
- Les scripts legacy dans `_old/legacy_scripts/` ne sont plus utilisés mais conservés pour référence
- La documentation a été consolidée dans les fichiers Markdown principaux
