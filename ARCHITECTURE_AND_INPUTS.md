# Architecture du Projet et Guide des Entrées Utilisateur

**Date:** 2024-01-18  
**Version:** 2.0

---

## 📋 Table des Matières

1. [Point d'Entrée Principal](#point-dentrée-principal)
2. [Architecture des Fichiers](#architecture-des-fichiers)
3. [Graphe de Dépendances](#graphe-de-dépendances)
4. [Flux de Données](#flux-de-données)
5. [Entrées Utilisateur (Inputs)](#entrées-utilisateur-inputs)
6. [Sorties Générées (Outputs)](#sorties-générées-outputs)
7. [Tests et Validation](#tests-et-validation)

---

## 🚀 Point d'Entrée Principal

### Fichier: `main.py`

**Rôle:** Point d'entrée unique du projet. Parse les arguments CLI et orchestre le pipeline.

**Flux d'exécution:**
```
main.py
  ↓
  parse_args() [src/app/cli.py]
  ↓
  args_to_config() [src/app/cli.py]
  ↓
  PipelineRunner(config) [src/app/pipeline_runner.py]
  ↓
  runner.run() → Exécute phases 1-5
```

**Importations principales:**
- `src.app.cli` → Parsing arguments
- `src.app.pipeline_runner` → Orchestration pipeline
- `src.config` → Configuration

---

## 🗂️ Architecture des Fichiers

### Structure Principale

```
ton_iot/
├── main.py                          # POINT D'ENTRÉE UNIQUE
├── src/
│   ├── app/                         # Application & CLI
│   │   ├── cli.py                   # Parsing arguments CLI
│   │   └── pipeline_runner.py       # Orchestrateur pipeline (5 phases)
│   │
│   ├── config.py                    # Configuration centralisée (PipelineConfig)
│   │
│   ├── core/                        # Modules core (traitement données)
│   │   ├── __init__.py              # Exports: DatasetLoader, DataHarmonizer, PreprocessingPipeline
│   │   ├── dataset_loader.py        # Chargement datasets (CIC-DDoS2019, TON_IoT)
│   │   ├── data_harmonization.py    # Harmonisation & early fusion
│   │   ├── preprocessing_pipeline.py # Pipeline preprocessing (cleaning, encoding, scaling, etc.)
│   │   ├── feature_engineering.py   # Feature engineering (CIC/TON spécifiques)
│   │   ├── model_utils.py           # Utilitaires modèles (fresh_model)
│   │   └── dependencies.py          # ⚠️ DEPRECATED (wrapper compatibilité)
│   │
│   ├── models/                      # Modèles ML
│   │   ├── __init__.py              # Exports: get_model_registry
│   │   ├── registry.py              # Registry modèles (LR, DT, RF, CNN, TabNet)
│   │   ├── sklearn_models.py        # Modèles sklearn (LR, DT, RF)
│   │   ├── cnn.py                   # CNN Tabular
│   │   └── tabnet.py                # TabNet wrapper
│   │
│   ├── phases/                      # 5 Phases du pipeline
│   │   ├── phase1_config_search.py  # Phase 1: Recherche 108 configs
│   │   ├── phase2_apply_best_config.py # Phase 2: Application meilleure config
│   │   ├── phase3_evaluation.py     # Phase 3: Évaluation 3D
│   │   ├── phase4_ahp_preferences.py # Phase 4: Préférences AHP
│   │   └── phase5_topsis_ranking.py # Phase 5: Ranking TOPSIS
│   │
│   ├── evaluation/                  # Évaluation & métriques
│   │   ├── visualizations.py        # Visualisations matplotlib-only
│   │   ├── metrics.py               # Calcul métriques (F1, accuracy, etc.)
│   │   ├── explainability.py        # SHAP, LIME, native interpretability
│   │   ├── resources.py             # Métriques ressources (temps, mémoire)
│   │   └── reporting.py             # Génération rapports
│   │
│   ├── evaluation_3d.py             # Framework évaluation 3D (dimensions)
│   │
│   ├── utils/                       # ⭐ NOUVEAU: Modules utilitaires
│   │   ├── path_helpers.py          # ensure_dir()
│   │   ├── optional_imports.py      # optional_import(), check_optional_import()
│   │   └── viz_helpers.py           # save_fig(), get_standard_colors()
│   │
│   ├── ahp_topsis_framework.py      # Framework AHP-TOPSIS
│   │
│   └── [DEPRECATED]
│       ├── main_pipeline.py         # ⚠️ DEPRECATED (remplacé par PipelineRunner)
│       ├── results_visualizer.py    # ⚠️ DEPRECATED (remplacé par evaluation/visualizations.py)
│       └── realtime_visualizer.py   # ⚠️ OPTIONNEL (GUI temps réel)
│
├── tests/                           # Tests pytest
│   ├── conftest.py                  # Fixtures pytest
│   ├── test_preprocessing_pipeline.py
│   ├── test_cnn.py
│   ├── test_tabnet.py
│   ├── test_no_data_leakage.py
│   ├── test_algo_handling.py
│   └── test_phase3_cnn_tabnet.py
│
└── datasets/                        # Datasets (input)
    ├── ton_iot/
    │   └── train_test_network.csv
    └── cic_ddos2019/
        └── [fichiers CSV]
```

---

## 🔗 Graphe de Dépendances

### Niveau 0: Point d'Entrée

```
main.py
  ↓ imports
  src/app/cli.py
  src/app/pipeline_runner.py
```

### Niveau 1: Application Layer

```
src/app/pipeline_runner.py
  ↓ imports
  src/config.py (PipelineConfig)
  src/phases/phase1_config_search.py
  src/phases/phase2_apply_best_config.py
  src/phases/phase3_evaluation.py
  src/phases/phase4_ahp_preferences.py
  src/phases/phase5_topsis_ranking.py

src/app/cli.py
  ↓ imports
  src/config.py (PipelineConfig)
```

### Niveau 2: Phases

```
phase1_config_search.py
  ↓ imports
  src/config.py
  src/core/__init__.py (DatasetLoader, DataHarmonizer, PreprocessingPipeline)

phase2_apply_best_config.py
  ↓ imports
  src/core/__init__.py
  src/core/feature_engineering.py

phase3_evaluation.py
  ↓ imports
  src/core/__init__.py
  src/core/feature_engineering.py
  src/core/model_utils.py
  src/core/preprocessing_pipeline.py
  src/models/__init__.py (get_model_registry)
  src/evaluation_3d.py
  src/models/cnn.py
  src/models/tabnet.py

phase4_ahp_preferences.py
  ↓ imports
  src/ahp_topsis_framework.py

phase5_topsis_ranking.py
  ↓ imports
  src/ahp_topsis_framework.py
```

### Niveau 3: Core Modules

```
src/core/dataset_loader.py
  ↓ imports
  src/system_monitor.py
  src/feature_analyzer.py
  src/irp_features_requirements.py

src/core/data_harmonization.py
  ↓ imports
  src/feature_analyzer.py
  src/irp_features_requirements.py

src/core/preprocessing_pipeline.py
  ↓ imports
  sklearn, imblearn, pandas, numpy (externes)

src/core/model_utils.py
  ↓ imports
  sklearn.base (externe)
```

### Niveau 4: Evaluation & Models

```
src/evaluation_3d.py
  ↓ imports
  src/core/__init__.py
  src/core/model_utils.py
  src/core/preprocessing_pipeline.py
  src/evaluation/visualizations.py
  src/evaluation/explainability.py
  src/evaluation/resources.py

src/models/registry.py
  ↓ imports
  src/models/sklearn_models.py
  src/models/cnn.py
  src/models/tabnet.py
```

### Niveau 5: Utils & Helpers

```
src/utils/path_helpers.py
  ↓ imports
  pathlib (standard)

src/utils/viz_helpers.py
  ↓ imports
  matplotlib (optionnel)

src/utils/optional_imports.py
  ↓ imports
  (aucun)
```

### Diagramme de Flux Complet

```
main.py
  │
  ├─→ cli.py ──────┐
  │                 │
  └─→ pipeline_runner.py ←──┘
        │
        ├─→ phase1_config_search.py
        │     ├─→ config.py
        │     ├─→ core/__init__.py
        │     │     ├─→ dataset_loader.py
        │     │     │     ├─→ system_monitor.py
        │     │     │     ├─→ feature_analyzer.py
        │     │     │     └─→ irp_features_requirements.py
        │     │     ├─→ data_harmonization.py
        │     │     │     ├─→ feature_analyzer.py
        │     │     │     └─→ irp_features_requirements.py
        │     │     └─→ preprocessing_pipeline.py
        │     │
        ├─→ phase2_apply_best_config.py
        │     ├─→ core/__init__.py
        │     └─→ core/feature_engineering.py
        │
        ├─→ phase3_evaluation.py
        │     ├─→ core/__init__.py
        │     ├─→ core/model_utils.py
        │     ├─→ models/__init__.py
        │     │     └─→ models/registry.py
        │     │           ├─→ models/sklearn_models.py
        │     │           ├─→ models/cnn.py
        │     │           └─→ models/tabnet.py
        │     ├─→ evaluation_3d.py
        │     │     ├─→ evaluation/visualizations.py
        │     │     ├─→ evaluation/explainability.py
        │     │     └─→ evaluation/resources.py
        │     └─→ core/feature_engineering.py
        │
        ├─→ phase4_ahp_preferences.py
        │     └─→ ahp_topsis_framework.py
        │
        └─→ phase5_topsis_ranking.py
              └─→ ahp_topsis_framework.py
```

---

## 📥 Entrées Utilisateur (Inputs)

### 1. Arguments CLI (via `main.py`)

#### Arguments Principaux

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--phase` | int | None (toutes) | Phase à exécuter (1-5) |
| `--output-dir` | str | `output` | Répertoire de sortie |
| `--test-mode` | flag | False | Mode test (0.1% données) |
| `--sample-ratio` | float | 1.0 | Ratio données (0.0-1.0) |
| `--cic-max-files` | int | None | Nombre max fichiers CIC-DDoS2019 |
| `--synthetic` | flag | False | Utiliser données synthétiques Phase 3 |
| `--interactive` | flag | False | Activer UI Tkinter |
| `--random-state` | int | 42 | Seed aléatoire |

#### Arguments Phase 4 (AHP)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ahp-dim1-weight` | float | None | Poids Dimension 1 (Performance) |
| `--ahp-dim2-weight` | float | None | Poids Dimension 2 (Resources) |
| `--ahp-dim3-weight` | float | None | Poids Dimension 3 (Explainability) |

**Contrainte:** `dim1 + dim2 + dim3 = 1.0` (normalisé automatiquement)

#### Exemples d'Utilisation

```bash
# Pipeline complet (production)
python main.py

# Pipeline complet (test mode, rapide)
python main.py --test-mode

# Phase spécifique
python main.py --phase 3

# Personnalisé
python main.py \
  --output-dir custom_results \
  --sample-ratio 0.01 \
  --cic-max-files 5 \
  --random-state 123

# Avec données synthétiques Phase 3
python main.py --synthetic --phase 3

# Avec préférences AHP personnalisées
python main.py \
  --ahp-dim1-weight 0.6 \
  --ahp-dim2-weight 0.2 \
  --ahp-dim3-weight 0.2
```

### 2. Fichiers de Configuration (via `PipelineConfig`)

#### Fichier: `src/config.py`

**Classe:** `PipelineConfig`

**Champs Configurables:**

```python
# Mode et chemins
test_mode: bool = False
sample_ratio: float = 1.0
random_state: int = 42
output_dir: str = "output"
cic_max_files: Optional[int] = None
synthetic_mode: bool = False

# Phase 1
phase1_search_enabled: bool = True
phase1_n_configs: int = 108

# Phase 3
phase3_enabled: bool = True
phase3_algorithms: List[str] = ['Logistic_Regression', 'Decision_Tree', 'Random_Forest', 'CNN', 'TabNet']
phase3_cv_folds: int = 5

# Preprocessing profiles (model-aware)
preprocessing_profiles: Dict[str, Dict] = {
    'lr_profile': {...},
    'tree_profile': {...},
    'cnn_profile': {...},
    'tabnet_profile': {...}
}

# Phase 4 (AHP)
ahp_preferences: Dict[str, float] = {
    'dimension1_performance': 0.5,
    'dimension2_resources': 0.3,
    'dimension3_explainability': 0.2
}

# Chemins datasets
dataset_paths: Dict[str, str] = {
    'ton_iot': 'datasets/ton_iot/train_test_network.csv',
    'cic_ddos2019': 'datasets/cic_ddos2019/'
}
```

**Modification:** Créer une instance `PipelineConfig` avec valeurs personnalisées et passer à `PipelineRunner`.

### 3. Fichiers Datasets (Inputs Externes)

#### TON_IoT Dataset

**Chemin:** `datasets/ton_iot/train_test_network.csv`

**Format:** CSV avec colonnes:
- Features numériques (multiple)
- `type` ou `label` (colonne cible)
- Format: CSV standard

**Chargement:**
- Via `DatasetLoader.load_ton_iot()`
- Sampling décimation si `sample_ratio < 1.0`

#### CIC-DDoS2019 Dataset

**Chemin:** `datasets/cic_ddos2019/`

**Format:** Multiple fichiers CSV
- Exemples: `DrDoS_LDAP.csv`, `DrDoS_MSSQL.csv`, `Syn.csv`, etc.
- Format: CSV avec colonne `Label` (BENIGN ou nom d'attaque)

**Chargement:**
- Via `DatasetLoader.load_cic_ddos2019()`
- Limité à `cic_max_files` fichiers (défaut: 3 en test mode, tous en production)
- Sampling décimation si `sample_ratio < 1.0`

### 4. Fichiers de Configuration Phase 1 (Optionnel)

**Chemin:** `output/phase1_config_search/best_config.json`

**Format:** JSON avec structure:
```json
{
  "config_id": 42,
  "apply_encoding": true,
  "apply_feature_selection": true,
  "feature_selection_k": 20,
  "apply_scaling": true,
  "scaling_method": "RobustScaler",
  "apply_resampling": true,
  "resampling_method": "SMOTE",
  "f1_score": 0.95
}
```

**Utilisation:** Phase 2 lit automatiquement ce fichier si Phase 1 a été exécutée.

---

## 📤 Sorties Générées (Outputs)

### Structure des Sorties

```
output/
├── logs/
│   └── main_YYYYMMDD_HHMMSS.log     # Logs généraux
│
├── phase1_config_search/
│   ├── best_config.json              # Meilleure configuration
│   └── config_evaluation_results.csv # Résultats 108 configs
│
├── phase2_apply_best_config/
│   ├── best_preprocessed.parquet     # Dataset préprocessé (ou .csv.gz)
│   └── preprocessing_stats.json      # Statistiques preprocessing
│
├── phase3_evaluation/
│   ├── evaluation_results.csv        # Métriques par algorithme
│   ├── dimension_scores.csv          # Scores 3D par algorithme
│   ├── metrics_by_fold.csv           # Métriques par fold CV
│   │
│   ├── algorithm_reports/
│   │   ├── LR_report.md
│   │   ├── DT_report.md
│   │   ├── RF_report.md
│   │   ├── CNN_report.md
│   │   └── TabNet_report.md
│   │
│   ├── metrics/
│   │   ├── mutual_information.csv
│   │   ├── permutation_importance.csv
│   │   ├── ratio_validation.json
│   │   └── INDEX.md
│   │
│   └── visualizations/
│       ├── [27 visualisations PNG]
│       └── INDEX.md
│
├── phase4_ahp_preferences/
│   └── ahp_weights.csv               # Poids AHP calculés
│
└── phase5_ranking/
    ├── ranking_results.csv           # Résultats ranking TOPSIS
    └── ranking_results.md            # Version markdown
```

---

## 🔄 Flux de Données

### Flux Principal

```
1. USER INPUT
   └─→ Arguments CLI (main.py)
       └─→ PipelineConfig

2. DATASETS INPUT
   ├─→ TON_IoT CSV
   └─→ CIC-DDoS2019 CSVs

3. PHASE 1: Config Search
   └─→ 108 configurations testées
       └─→ best_config.json

4. PHASE 2: Apply Best Config
   ├─→ Lecture best_config.json
   ├─→ Harmonisation datasets
   ├─→ Feature engineering (stateless)
   └─→ best_preprocessed.parquet

5. PHASE 3: Evaluation
   ├─→ Lecture best_preprocessed.parquet
   ├─→ Cross-validation (5 folds)
   ├─→ Preprocessing model-aware par fold
   ├─→ Évaluation 5 algorithmes (LR, DT, RF, CNN, TabNet)
   ├─→ Calcul métriques 3D (Performance, Resources, Explainability)
   └─→ Génération rapports + visualisations

6. PHASE 4: AHP Preferences
   ├─→ Lecture dimension_scores.csv
   ├─→ Calcul poids AHP (ou utilisation poids CLI)
   └─→ ahp_weights.csv

7. PHASE 5: TOPSIS Ranking
   ├─→ Lecture dimension_scores.csv + ahp_weights.csv
   ├─→ Calcul scores TOPSIS
   └─→ ranking_results.csv + .md
```

### Flux de Preprocessing (Phase 3)

```
Pour chaque algorithme:
  Pour chaque fold CV:
    1. Split TRAIN / TEST
    2. FIT preprocessing sur TRAIN uniquement:
       - Imputation (median)
       - Feature selection (si activé)
       - Scaling (si activé, selon profile)
    3. TRANSFORM TEST (stateless):
       - Imputer.transform()
       - FeatureSelector.transform()
       - Scaler.transform() (si activé)
    4. Resampling (SMOTE) sur TRAIN uniquement
    5. Train model sur TRAIN préprocessé
    6. Evaluate sur TEST préprocessé
```

---

## 🧪 Tests et Validation

### Structure des Tests

```
tests/
├── conftest.py                      # Fixtures pytest communes
├── test_preprocessing_pipeline.py   # Tests preprocessing
├── test_cnn.py                      # Tests CNN
├── test_tabnet.py                   # Tests TabNet
├── test_no_data_leakage.py          # Tests zéro data leakage
├── test_algo_handling.py            # Tests helpers algo
├── test_phase3_cnn_tabnet.py        # Tests Phase 3 CNN/TabNet
└── _legacy_tests/                   # Tests legacy (exclus par pytest.ini)
```

### Exécution des Tests

```bash
# Tous les tests
.toniot/bin/python -m pytest tests/ -v

# Test spécifique
.toniot/bin/python -m pytest tests/test_preprocessing_pipeline.py -v

# Avec coverage
.toniot/bin/python -m pytest tests/ --cov=src --cov-report=html
```

### Tests Clés

| Test | Objectif |
|------|----------|
| `test_transform_test_requires_fitted_pipeline` | Vérifie que pipeline doit être fitted avant transform_test() |
| `test_no_data_leakage` | Vérifie zéro data leakage (scaler/selector/imputer fit uniquement sur TRAIN) |
| `test_cnn_pipeline_full_flow` | Test intégration complète (5 phases) avec CNN |
| `test_tabnet_pipeline_full_flow` | Test intégration complète (5 phases) avec TabNet |
| `test_algo_handling` | Test helpers algo (get_algo_names, sanitize_algo_name) |

---

## 📊 Liens Entre Modules Clés

### 1. Core → Phases

```
core/__init__.py
  ├─→ DatasetLoader          → Utilisé par phase1, phase2, phase3
  ├─→ DataHarmonizer         → Utilisé par phase1, phase2, phase3
  └─→ PreprocessingPipeline  → Utilisé par phase1, phase2, phase3
```

### 2. Models → Phase 3

```
models/registry.py
  ├─→ get_model_registry()   → Utilisé par phase3_evaluation.py
  ├─→ sklearn_models.py      → LR, DT, RF builders
  ├─→ cnn.py                 → CNN builder
  └─→ tabnet.py              → TabNet builder
```

### 3. Evaluation → Phase 3

```
evaluation_3d.py
  ├─→ Evaluation3D.evaluate_model() → Utilisé par phase3_evaluation.py
  ├─→ evaluation/visualizations.py  → Génération visu
  ├─→ evaluation/explainability.py  → SHAP, LIME
  └─→ evaluation/resources.py       → Métriques ressources
```

### 4. Utils → Multiple Modules

```
utils/path_helpers.py
  └─→ ensure_dir()           → Utilisable partout (pas encore utilisé partout)

utils/viz_helpers.py
  └─→ save_fig()             → Utilisable pour visualisations

utils/optional_imports.py
  └─→ optional_import()      → Utilisable pour imports optionnels
```

---

## 🔍 Points d'Entrée Détaillés

### 1. Point d'Entrée Principal: `main.py`

**Signature:**
```python
def main() -> int:
    """Main entry point"""
    # 1. Parse CLI arguments
    # 2. Create PipelineConfig
    # 3. Setup logging
    # 4. Initialize PipelineRunner
    # 5. Execute pipeline
    # 6. Return exit code
```

**Inputs:**
- Arguments CLI (via `sys.argv`)
- Fichiers datasets (via chemins dans `config.dataset_paths`)

**Outputs:**
- Exit code (0 = succès, 1 = erreur, 130 = interrupt)
- Fichiers résultats dans `output/`

### 2. Point d'Entrée API: `PipelineRunner`

**Signature:**
```python
class PipelineRunner:
    def __init__(self, config: PipelineConfig):
        """Initialize with configuration"""
    
    def run(self, phases: Optional[list] = None) -> Dict:
        """Run pipeline phases"""
```

**Inputs:**
- `config: PipelineConfig` (configuration complète)
- `phases: Optional[list]` (liste phases à exécuter, ou None = toutes)

**Outputs:**
- `Dict[int, Any]` (résultats par phase)

**Utilisation:**
```python
from src.config import PipelineConfig
from src.app.pipeline_runner import PipelineRunner

config = PipelineConfig(
    test_mode=True,
    sample_ratio=0.01,
    output_dir="custom_output"
)
runner = PipelineRunner(config)
results = runner.run(phases=[1, 2, 3])
```

---

## 🎯 Résumé des Inputs Utilisateur

### Inputs Requis

1. **Datasets** (optionnel si `--synthetic`):
   - `datasets/ton_iot/train_test_network.csv`
   - `datasets/cic_ddos2019/*.csv`

2. **Arguments CLI** (via `main.py`):
   - Minimum: Aucun (défauts utilisés)
   - Recommandé: `--test-mode` pour premiers tests

### Inputs Optionnels

1. **Configuration personnalisée**:
   - Modifier `PipelineConfig` dans code
   - Ou utiliser arguments CLI

2. **Fichier best_config.json** (si Phase 1 sautée):
   - Placer dans `output/phase1_config_search/best_config.json`

3. **AHP Preferences** (Phase 4):
   - Via arguments CLI `--ahp-dim1-weight`, etc.
   - Ou fichier `output/phase4_ahp_preferences/ahp_weights.csv`

---

## 📝 Notes Importantes

### ⚠️ Dependencies.py est DEPRECATED

**Ancien code:**
```python
from src.core.dependencies import np, pd, Path
```

**Nouveau code (recommandé):**
```python
import numpy as np
import pandas as pd
from pathlib import Path
```

### ⚠️ Main_pipeline.py est DEPRECATED

**Ancien code:**
```python
from src.main_pipeline import IRPPipeline
pipeline = IRPPipeline()
```

**Nouveau code:**
```python
from src.app.pipeline_runner import PipelineRunner
from src.config import PipelineConfig
config = PipelineConfig()
runner = PipelineRunner(config)
results = runner.run()
```

### ⚠️ Results_visualizer.py est DEPRECATED

**Ancien code:**
```python
from src.results_visualizer import ResultsVisualizer
app = ResultsVisualizer(output_dir)
```

**Nouveau code:**
```python
from src.evaluation.visualizations import generate_all_visualizations
generate_all_visualizations(metrics_df, output_dir=output_dir)
```

---

## 🔗 Liens Externes

### Documentation

- **README.md** : Guide utilisateur principal
- **src/README.md** : Documentation modules src/
- **tests/TESTS_DOCUMENTATION.md** : Documentation tests
- **PROJECT_STRUCTURE.md** : Structure projet détaillée

### Rapports

- **output/static_analysis/STATIC_ANALYSIS_REPORT.md** : Analyse statique complète
- **output/static_analysis/REPORT.md** : Rapport statique (résumé)

---

**Document généré le:** 2024-01-18  
**Dernière mise à jour:** Après optimisations (découplage dependencies.py, factorisation, utils modules)
