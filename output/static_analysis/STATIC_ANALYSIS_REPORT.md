# Analyse Statique Complète - TON_IoT Project

**Date:** 2024-01-18  
**Branch:** dev2  
**Total fichiers Python:** 40  
**Total lignes:** ~11,359  

---

## 📊 Résumé Exécutif

### Métriques Globales
- **Fichiers Python:** 40
- **Modules principaux:** core (7), phases (5), models (5), evaluation (6), app (3)
- **Fichiers isolés/morts:** 3-4 identifiés
- **Cycles d'imports:** 1 potentiel (core/dependencies)
- **Redondances majeures:** 5+ identifiées
- **Duplications:** ~10 patterns identifiés

### Priorités d'Optimisation
1. **HIGH:** Découpler `core/dependencies.py` (god module)
2. **HIGH:** Supprimer code mort (`main_pipeline.py`, `results_visualizer.py`)
3. **MEDIUM:** Factoriser duplication transform_test / _transform_test_fold
4. **MEDIUM:** Consolider visualisations (3 modules séparés)
5. **LOW:** Extraire helpers communs (algo naming, path utils)

---

## 🗺️ Graphe Logique des Modules

### Architecture Globale

```
main.py (entrypoint)
    └── app/pipeline_runner.py
            ├── phases/phase1_config_search.py
            ├── phases/phase2_apply_best_config.py
            ├── phases/phase3_evaluation.py
            ├── phases/phase4_ahp_preferences.py
            └── phases/phase5_topsis_ranking.py

phases/
    ├── phase3_evaluation.py ──┐
    └── phase2_apply_best_config.py ──┐
            │                          │
            └──────────────────────────┴──→ core/
                                               ├── dataset_loader.py
                                               ├── data_harmonization.py
                                               ├── preprocessing_pipeline.py
                                               └── __init__.py

core/
    ├── dependencies.py ──┐
    │                     │ (god module)
    └─────────────────────┘
            ├──→ system_monitor.py
            ├──→ feature_analyzer.py
            └──→ irp_features_requirements.py

evaluation/
    ├── visualizations.py ──→ (matplotlib-only)
    ├── metrics.py
    ├── explainability.py
    └── resources.py

evaluation_3d.py ──→ evaluation/visualizations.py

models/
    ├── cnn.py
    ├── tabnet.py
    ├── sklearn_models.py
    └── registry.py

ISOLÉS (code mort):
    ├── main_pipeline.py (IRPPipeline - jamais utilisé)
    ├── results_visualizer.py (seulement verify_irp_compliance.py)
    └── realtime_visualizer.py (optionnel, main_pipeline.py seulement)
```

### Détails des Imports par Module

#### Modules Centraux (God Modules)

1. **`core/dependencies.py`** ⚠️ PROBLÉMATIQUE
   - **Imports:** system_monitor, feature_analyzer, irp_features_requirements
   - **Exports:** np, pd, Path, SystemMonitor, FeatureAnalyzer, IRPFeaturesRequirements, etc.
   - **Utilisé par:** dataset_loader.py, data_harmonization.py, main_pipeline.py
   - **Problème:** Centralise trop d'imports, crée couplage fort

2. **`core/__init__.py`** ✅ ACCEPTABLE
   - **Exports:** DatasetLoader, DataHarmonizer, PreprocessingPipeline, StratifiedCrossValidator
   - **Utilisé par:** phases/*, app/*, main_pipeline.py
   - **Statut:** Bon découplage, exports principaux

3. **`config.py`** ✅ EXCELLENT
   - **Imports:** Minimal (typing, dataclasses)
   - **Exports:** PipelineConfig, generate_108_configs, TEST_CONFIG
   - **Utilisé par:** Tous les phases, app/cli.py
   - **Statut:** Pas de dépendances, très découplé

#### Modules Phases (5)

- **`phases/phase1_config_search.py`**
  - Imports: `config`, `core` (via __init__.py)
  - Utilisé par: `app/pipeline_runner.py`

- **`phases/phase2_apply_best_config.py`**
  - Imports: `core`, `core.feature_engineering`
  - Utilisé par: `app/pipeline_runner.py`

- **`phases/phase3_evaluation.py`** 🔥 HOTSPOT
  - Imports: `core`, `models.cnn`, `models.tabnet`, `evaluation_3d`
  - Utilisé par: `app/pipeline_runner.py`
  - Problème: Méthode `_transform_test_fold()` duplique `transform_test()`

- **`phases/phase4_ahp_preferences.py`**
  - Imports: `ahp_topsis_framework`
  - Utilisé par: `app/pipeline_runner.py`

- **`phases/phase5_topsis_ranking.py`**
  - Imports: `ahp_topsis_framework`
  - Utilisé par: `app/pipeline_runner.py`

#### Modules Evaluation (6)

- **`evaluation/visualizations.py`** ✅ MATPLOTLIB-ONLY
  - Imports: matplotlib uniquement
  - Exports: `generate_all_visualizations()`, helpers algo
  - Utilisé par: `evaluation_3d.py`

- **`evaluation/metrics.py`**
  - Imports: sklearn.metrics
  - Utilisé par: Potentiellement phases/*

- **`evaluation/explainability.py`**
  - Imports: shap, lime (optionnels)
  - Utilisé par: `evaluation_3d.py`

- **`evaluation/resources.py`**
  - Imports: psutil, time
  - Utilisé par: Potentiellement phases/*

- **`evaluation_3d.py`** 🔥 HOTSPOT
  - Imports: `core`, `evaluation.visualizations`, sklearn
  - Utilisé par: `phases/phase3_evaluation.py`, `main_pipeline.py` (code mort)
  - Problème: Fonction très longue `evaluate_model()` (~170 lignes)

#### Modules Models (5)

- **`models/cnn.py`**
  - Imports: torch (optionnel)
  - Utilisé par: `phases/phase3_evaluation.py`, `models/registry.py`

- **`models/tabnet.py`**
  - Imports: pytorch_tabnet (optionnel)
  - Utilisé par: `phases/phase3_evaluation.py`, `models/registry.py`

- **`models/sklearn_models.py`**
  - Imports: sklearn
  - Utilisé par: `models/registry.py`

- **`models/registry.py`**
  - Imports: models/cnn, models/tabnet, models/sklearn_models
  - Utilisé par: Potentiellement phases/*

---

## 🔄 Cycles d'Imports Détectés

### Cycle Potentiel #1: core/dependencies.py ↔ core modules

**Cycle détecté:**
```
core/dependencies.py 
    ↓ (imports)
system_monitor.py, feature_analyzer.py, irp_features_requirements.py
    ↓ (utilisés par)
core/dataset_loader.py, core/data_harmonization.py
    ↓ (imports)
core/dependencies.py (via __all__ exports)
```

**Impact:**
- **Couplage:** Fort couplage via `dependencies.py`
- **Bugs potentiels:** ImportError si ordre d'import incorrect
- **Lenteur:** Import de tous les modules au chargement
- **Tests:** Instabilité potentielle selon ordre d'exécution

**Analyse:**
- ✅ **Pas de cycle direct:** `system_monitor.py` n'importe pas `core/dependencies.py`
- ⚠️ **Couplage indirect:** Tous les modules core importent `dependencies.py` qui importe d'autres modules
- 🔴 **God module:** `dependencies.py` centralise trop d'exports

**Solution proposée:**
1. Découpler `dependencies.py` en modules spécialisés:
   - `src/utils/type_hints.py` (typing)
   - `src/utils/path_helpers.py` (Path)
   - `src/utils/numpy_helpers.py` (np, pd si vraiment nécessaire)
2. Imports directs dans chaque module au lieu de passer par `dependencies.py`
3. Garder `dependencies.py` uniquement pour exports de compatibilité (deprecated)

### Cycle Potentiel #2: core/__init__.py ↔ core/*.py

**Cycle détecté:**
```
core/__init__.py
    ↓ (imports)
dataset_loader.py, data_harmonization.py, preprocessing_pipeline.py
    ↓ (chacun imports)
core/dependencies.py
    ↓ (pas de cycle direct car __init__.py n'importe pas dependencies.py)
```

**Impact:**
- ✅ **Pas de cycle réel:** `__init__.py` n'importe pas `dependencies.py`
- ⚠️ **Couplage via `dependencies.py`:** Tous dépendent du god module

**Solution proposée:**
- Découpler `dependencies.py` (comme ci-dessus) résoudra ce couplage

---

## 🔄 Redondances de Code

### 1. Duplication: `transform_test()` vs `_transform_test_fold()`

**Fichiers concernés:**
- `src/core/preprocessing_pipeline.py:738` → `transform_test()` (58 lignes)
- `src/phases/phase3_evaluation.py:373` → `_transform_test_fold()` (31 lignes)

**Analyse:**
- **Similarité:** ~80% de logique identique
- **Différences:**
  - `transform_test()`: Utilise `sanitize_numeric_values()`, gestion robuste
  - `_transform_test_fold()`: Logique manuelle, moins robuste
- **Problème:** Phase 3 devrait utiliser `transform_test()` directement

**Solution:**
```python
# Dans phase3_evaluation.py, remplacer:
X_test_prep = self._transform_test_fold(pipeline, X_test_fold, profile)

# Par:
X_test_prep = pipeline.transform_test(X_test_fold)
```

**Impact:** Réduire duplication, garantir cohérence, simplification

### 2. Duplication: Helpers Algo Names

**Fichiers concernés:**
- `src/evaluation/visualizations.py:117` → `get_algo_names()`
- `src/evaluation/visualizations.py:142` → `ensure_algo_column()`
- `src/evaluation/visualizations.py:169` → `sanitize_algo_name()`

**Statut:** ✅ **Déjà factorisé** (récent)

### 3. Duplication: Visualisations (3 modules séparés)

**Fichiers concernés:**
- `src/evaluation/visualizations.py` (783 lignes, matplotlib-only)
- `src/realtime_visualizer.py` (649 lignes, tkinter + seaborn)
- `src/results_visualizer.py` (842 lignes, tkinter)

**Analyse:**
- **Fonctions similaires:** `save_fig()`, création de figures matplotlib
- **Overlap:** Bar charts, scatter plots, heatmaps
- **Problème:** 3 modules avec logique de visualisation similaire

**Redondances identifiées:**
1. **Sauvegarde de figures:**
   - `visualizations.py:34` → `save_fig()` (matplotlib)
   - `realtime_visualizer.py` → `fig.savefig()` (inline)
   - `results_visualizer.py` → `FigureCanvasTkAgg` (tkinter)

2. **Création de bar charts:**
   - Présent dans les 3 modules avec logique similaire

3. **Gestion des couleurs/styles:**
   - Chaque module définit ses propres color schemes

**Solution proposée:**
- Créer `src/utils/viz_helpers.py` avec:
  - `save_fig()` (unifié)
  - `create_bar_chart()`, `create_scatter_plot()`, etc.
  - Color schemes standardisés
- `visualizations.py` reste matplotlib-only (Phase 3)
- `realtime_visualizer.py` et `results_visualizer.py` utilisent helpers communs

### 4. Duplication: Preprocessing Logic

**Fichiers concernés:**
- `src/core/preprocessing_pipeline.py:136` → `clean_data()`
- `src/core/preprocessing_pipeline.py:678` → `transform_data()`
- `src/phases/phase2_apply_best_config.py:143` → `_apply_stateless_preprocessing()`
- `src/phases/phase3_evaluation.py:338` → `_apply_preprocessing_per_fold()`

**Analyse:**
- **Similarité:** Logique de preprocessing répétée avec variations
- **Problème:** Chaque phase réimplémente des parties de preprocessing

**Statut:** ✅ **Acceptable** (workflow par phase avec profils différents)

### 5. Duplication: Model Building

**Fichiers concernés:**
- `src/phases/phase3_evaluation.py:405` → `_build_models()`
- `src/models/registry.py:21` → `get_model_registry()`

**Analyse:**
- **Similarité:** Construction de modèles avec try/except pour optionnels
- **Problème:** Deux endroits pour construire les mêmes modèles

**Solution proposée:**
- Phase 3 devrait utiliser `get_model_registry(config)` au lieu de `_build_models()`

### 6. Duplication: Path/File Utilities

**Patterns répétés:**
- `Path(output_dir).mkdir(parents=True, exist_ok=True)` (répété ~20 fois)
- `output_dir / 'phase3_evaluation' / 'file.csv'` (pattern répété)

**Solution proposée:**
- Créer `src/utils/path_helpers.py`:
  ```python
  def ensure_dir(path: Path) -> Path:
      path.mkdir(parents=True, exist_ok=True)
      return path
  ```

### 7. Duplication: Try/Except Imports

**Patterns répétés:**
- Try/except pour matplotlib, shap, lime, torch (répété ~10 fois)

**Solution proposée:**
- Créer `src/utils/optional_imports.py`:
  ```python
  def optional_import(module_name: str, default=None):
      try:
          return __import__(module_name), True
      except ImportError:
          return default, False
  ```

---

## 💀 Fichiers Morts ou Quasi-Inutilisés

### 1. `src/main_pipeline.py` - CODE MORT CONFIRMÉ 🔴

**Analyse:**
- **Classe:** `IRPPipeline` (762 lignes)
- **Usage:** Aucun import dans le code principal
- **Seul usage:** `verify_irp_compliance.py` (script legacy)
- **Remplacé par:** `app/pipeline_runner.py` (PipelineRunner)

**Vérification:**
```bash
# Aucun import de main_pipeline dans src/
grep -r "main_pipeline\|IRPPipeline" src/ --exclude-dir=__pycache__
# Résultat: Seulement dans main_pipeline.py lui-même
```

**Recommandation:**
- **Option 1:** SUPPRIMER si `verify_irp_compliance.py` n'est pas critique
- **Option 2:** DÉPLACER vers `tests/_legacy_tests/` ou `scripts/legacy/`
- **Option 3:** DOCUMENTER comme deprecated/compatibilité

**Impact:** 
- **Lignes de code:** -762 lignes
- **Maintenance:** Réduit confusion architecturale
- **Risque:** Bas (non utilisé)

### 2. `src/results_visualizer.py` - CODE MORT PROBABLE 🟡

**Analyse:**
- **Classe:** `ResultsVisualizer` (842 lignes)
- **Usage:** Seulement dans `verify_irp_compliance.py` (legacy)
- **Remplacement:** `evaluation/visualizations.py` (matplotlib-only, headless)

**Vérification:**
```bash
# Aucun import de results_visualizer dans src/
grep -r "results_visualizer\|ResultsVisualizer" src/ --exclude-dir=__pycache__
# Résultat: Aucun (sauf verify_irp_compliance.py)
```

**Recommandation:**
- **Option 1:** SUPPRIMER si GUI tkinter non nécessaire
- **Option 2:** DÉPLACER vers `src/ui/` si GUI nécessaire
- **Option 3:** MARKER comme deprecated

**Impact:**
- **Lignes de code:** -842 lignes
- **Maintenance:** Réduit surface d'attaque
- **Risque:** Moyen (GUI optionnel)

### 3. `src/realtime_visualizer.py` - OPTIONNEL 🟡

**Analyse:**
- **Classe:** `RealTimeVisualizer` (649 lignes)
- **Usage:** Seulement dans `main_pipeline.py` (code mort) avec try/except
- **Statut:** Optionnel, GUI interactive

**Vérification:**
```bash
grep -r "realtime_visualizer\|RealTimeVisualizer" src/ --exclude-dir=__pycache__
# Résultat: main_pipeline.py (code mort), import optionnel
```

**Recommandation:**
- **Option 1:** GARDER (optionnel, peut être utile pour debugging)
- **Option 2:** DÉPLACER vers `src/ui/` pour cohérence
- **Option 3:** MARKER comme deprecated si non utilisé

**Impact:**
- **Maintenance:** Bas (optionnel)
- **Risque:** Bas (non utilisé par pipeline principal)

### 4. `src/system_monitor.py` - SOUS-UTILISÉ 🟢

**Analyse:**
- **Classe:** `SystemMonitor` (200 lignes)
- **Usage:** `main_pipeline.py` (code mort), `dataset_loader.py` (conditionnel)
- **Statut:** Utile mais sous-utilisé

**Recommandation:**
- **GARDER** mais optimiser usage
- Potentiellement utile pour monitoring RAM/CPU

**Impact:**
- **Maintenance:** Acceptable
- **Risque:** Bas (optionnel)

### 5. `src/feature_analyzer.py` - UTILISÉ ✅

**Analyse:**
- **Classe:** `FeatureAnalyzer` (536 lignes)
- **Usage:** `dataset_loader.py`, `data_harmonization.py`
- **Statut:** Actif, utilisé

**Recommandation:**
- **GARDER** (utile pour feature mapping intelligent)

---

## 📋 Tableau: Fichier | Rôle | Problèmes | Action Recommandée

| Fichier | Rôle | Problèmes | Action Recommandée | Impact |
|---------|------|-----------|-------------------|---------|
| `core/dependencies.py` | God module, exports centralisés | Couplage fort, import lourd | **Découpler** en utils spécialisés | HIGH |
| `main_pipeline.py` | Ancien pipeline orchestrateur | Code mort, jamais utilisé | **SUPPRIMER** ou déplacer legacy | HIGH |
| `results_visualizer.py` | GUI Tkinter pour résultats | Code mort (seulement legacy script) | **SUPPRIMER** ou déplacer ui/ | HIGH |
| `realtime_visualizer.py` | Visualisation temps réel | Optionnel, peu utilisé | **GARDER** ou déplacer ui/ | MEDIUM |
| `phase3_evaluation.py` | Évaluation 3D Phase 3 | `_transform_test_fold()` duplique `transform_test()` | **Factoriser** utiliser `transform_test()` | MEDIUM |
| `evaluation/visualizations.py` | Visualisations Phase 3 | Overlap avec realtime/results_visualizer | **Consolider** helpers communs | MEDIUM |
| `realtime_visualizer.py` | Visualisations temps réel | Logique matplotlib dupliquée | **Extraire** helpers communs | MEDIUM |
| `results_visualizer.py` | GUI résultats | Logique matplotlib dupliquée | **Extraire** helpers communs | MEDIUM |
| `core/dataset_loader.py` | Chargement datasets | Fonction très longue `load_cic_ddos2019()` | **Découper** en méthodes privées | MEDIUM |
| `evaluation_3d.py` | Framework évaluation 3D | Fonction très longue `evaluate_model()` | **Découper** en méthodes privées | MEDIUM |
| `system_monitor.py` | Monitoring système | Sous-utilisé | **GARDER** (utile) | LOW |
| `feature_analyzer.py` | Analyse features | Utilisé activement | **GARDER** | LOW |

---

## 🎯 Plan d'Optimisation Priorisé (10 Points Max)

### Priorité HIGH 🔴

#### 1. Découpler `core/dependencies.py`
**Fichiers:** `src/core/dependencies.py`  
**Action:** 
- Créer `src/utils/` avec modules spécialisés:
  - `type_hints.py` (typing exports)
  - `path_helpers.py` (Path, ensure_dir)
  - `numpy_helpers.py` (np, pd si nécessaire)
- Remplacer imports `from src.core.dependencies import ...` par imports directs
- Garder `dependencies.py` comme deprecated wrapper (compatibilité)

**Effort:** M (Medium)  
**Impact:** Réduit couplage, améliore testabilité, évite cycles

#### 2. Supprimer Code Mort - `main_pipeline.py`
**Fichiers:** `src/main_pipeline.py`  
**Action:**
- Vérifier usage dans `verify_irp_compliance.py`
- Si utilisé uniquement pour vérification: **DÉPLACER** vers `scripts/legacy/`
- Sinon: **SUPPRIMER**

**Effort:** S (Small)  
**Impact:** -762 lignes, réduction confusion

#### 3. Supprimer/Isoler `results_visualizer.py`
**Fichiers:** `src/results_visualizer.py`  
**Action:**
- Vérifier si GUI tkinter nécessaire
- Si non: **SUPPRIMER**
- Si oui: **DÉPLACER** vers `src/ui/results_visualizer.py`

**Effort:** S (Small)  
**Impact:** -842 lignes si supprimé

### Priorité MEDIUM 🟡

#### 4. Factoriser `transform_test()` vs `_transform_test_fold()`
**Fichiers:** `src/phases/phase3_evaluation.py`  
**Action:**
- Supprimer `_transform_test_fold()`
- Utiliser directement `pipeline.transform_test()` dans Phase 3
- Vérifier compatibilité avec profiles

**Effort:** S (Small)  
**Impact:** Réduction duplication, cohérence garantie

#### 5. Consolider Visualisations - Helpers Communs
**Fichiers:** `src/evaluation/visualizations.py`, `src/realtime_visualizer.py`, `src/results_visualizer.py`  
**Action:**
- Créer `src/utils/viz_helpers.py`:
  - `save_fig()` (unifié)
  - `create_bar_chart()`, `create_scatter_plot()` (helpers)
  - Color schemes standardisés
- Refactorer les 3 modules pour utiliser helpers communs

**Effort:** M (Medium)  
**Impact:** Réduction duplication, maintenance simplifiée

#### 6. Simplifier `DatasetLoader.load_cic_ddos2019()`
**Fichiers:** `src/core/dataset_loader.py`  
**Action:**
- Découper en méthodes privées:
  - `_load_cic_file()`
  - `_validate_cic_data()`
  - `_chunk_cic_data()`
- Réduire complexité cyclomatique

**Effort:** M (Medium)  
**Impact:** Meilleure lisibilité, testabilité

#### 7. Simplifier `Evaluation3D.evaluate_model()`
**Fichiers:** `src/evaluation_3d.py`  
**Action:**
- Extraire méthodes privées:
  - `_compute_dimension_1_performance()`
  - `_compute_dimension_2_resources()`
  - `_compute_dimension_3_explainability()`

**Effort:** M (Medium)  
**Impact:** Meilleure lisibilité, testabilité

#### 8. Utiliser `get_model_registry()` dans Phase 3
**Fichiers:** `src/phases/phase3_evaluation.py`  
**Action:**
- Remplacer `_build_models()` par `get_model_registry(config)`
- Réduire duplication avec `models/registry.py`

**Effort:** S (Small)  
**Impact:** Cohérence, source unique de vérité

### Priorité LOW 🟢

#### 9. Extraire Helpers Optionnels
**Fichiers:** Tous avec try/except imports  
**Action:**
- Créer `src/utils/optional_imports.py`:
  ```python
  def optional_import(module_name: str, default=None):
      try:
          return __import__(module_name), True
      except ImportError:
          return default, False
  ```

**Effort:** S (Small)  
**Impact:** Réduction duplication try/except

#### 10. Extraire Path Utilities
**Fichiers:** Tous avec `Path(...).mkdir(...)`  
**Action:**
- Créer `src/utils/path_helpers.py`:
  ```python
  def ensure_dir(path: Path) -> Path:
      path.mkdir(parents=True, exist_ok=True)
      return path
  ```

**Effort:** S (Small)  
**Impact:** Réduction duplication, cohérence

---

## 🔍 Focus Spécial Tests

### Analyse des Imports Tests

**Structure tests:**
- `tests/` (actifs, collectés par pytest)
- `tests/_legacy_tests/` (exclus par pytest.ini)

**Cohérence imports:**
- ✅ Tous les tests utilisent `sys.path.insert(0, project_root)`
- ✅ Pattern uniforme: `from src.core.preprocessing_pipeline import ...`
- ✅ Pas d'imports circulaires tests ↔ core

**Doublons de tests:**
- `tests/test_no_data_leakage.py` (320 lignes) ✅ ACTIF
- `tests/_legacy_tests/test_no_data_leakage.py` (55 lignes) ❌ IGNORÉ
- **Statut:** Conflit résolu via pytest.ini (norecursedirs)

**Tests couvrant même logique:**
- `test_preprocessing_pipeline.py` → Tests sanitize, transform_test
- `test_no_data_leakage.py` → Tests scaler/selector/imputer fit uniquement sur TRAIN
- **Statut:** Couverture complémentaire, pas de doublons

---

## 📊 Graphe d'Imports Détail

### Modules Core (7 fichiers)

```
core/
├── __init__.py
│   ├── imports: dataset_loader, data_harmonization, preprocessing_pipeline
│   └── exports: DatasetLoader, DataHarmonizer, PreprocessingPipeline, StratifiedCrossValidator
│
├── dependencies.py ⚠️ GOD MODULE
│   ├── imports: system_monitor, feature_analyzer, irp_features_requirements
│   └── exports: np, pd, Path, SystemMonitor, FeatureAnalyzer, IRPFeaturesRequirements, ...
│
├── dataset_loader.py
│   ├── imports: dependencies (→ system_monitor, feature_analyzer)
│   └── utilisé par: phases/*, main_pipeline.py (mort)
│
├── data_harmonization.py
│   ├── imports: dependencies (→ feature_analyzer, irp_features_requirements)
│   └── utilisé par: phases/*, main_pipeline.py (mort)
│
├── preprocessing_pipeline.py
│   ├── imports: sklearn, imblearn, pandas, numpy
│   └── utilisé par: phases/*, evaluation_3d.py
│
├── feature_engineering.py
│   ├── imports: pandas, numpy
│   └── utilisé par: phases/phase2, phases/phase3
│
└── model_utils.py
    ├── imports: sklearn.base
    └── utilisé par: phases/phase3, evaluation_3d.py
```

### Modules Phases (5 fichiers)

```
phases/
├── phase1_config_search.py
│   ├── imports: config, core
│   └── utilisé par: app/pipeline_runner.py
│
├── phase2_apply_best_config.py
│   ├── imports: core, core.feature_engineering
│   └── utilisé par: app/pipeline_runner.py
│
├── phase3_evaluation.py 🔥 HOTSPOT
│   ├── imports: core, models.cnn, models.tabnet, evaluation_3d
│   ├── problèmes: _transform_test_fold() duplique transform_test()
│   └── utilisé par: app/pipeline_runner.py
│
├── phase4_ahp_preferences.py
│   ├── imports: ahp_topsis_framework
│   └── utilisé par: app/pipeline_runner.py
│
└── phase5_topsis_ranking.py
    ├── imports: ahp_topsis_framework
    └── utilisé par: app/pipeline_runner.py
```

### Modules Evaluation (6 fichiers)

```
evaluation/
├── visualizations.py ✅ MATPLOTLIB-ONLY
│   ├── imports: matplotlib uniquement
│   ├── exports: generate_all_visualizations(), helpers algo
│   └── utilisé par: evaluation_3d.py
│
├── metrics.py
│   ├── imports: sklearn.metrics
│   └── utilisé par: Potentiellement phases/*
│
├── explainability.py
│   ├── imports: shap, lime (optionnels)
│   └── utilisé par: evaluation_3d.py
│
└── resources.py
    ├── imports: psutil, time
    └── utilisé par: Potentiellement phases/*

evaluation_3d.py 🔥 HOTSPOT
├── imports: core, evaluation.visualizations, sklearn
├── problèmes: evaluate_model() très long (~170 lignes)
└── utilisé par: phases/phase3_evaluation.py, main_pipeline.py (mort)
```

### Modules Models (5 fichiers)

```
models/
├── cnn.py
│   ├── imports: torch (optionnel)
│   └── utilisé par: phases/phase3_evaluation.py, models/registry.py
│
├── tabnet.py
│   ├── imports: pytorch_tabnet (optionnel)
│   └── utilisé par: phases/phase3_evaluation.py, models/registry.py
│
├── sklearn_models.py
│   ├── imports: sklearn
│   └── utilisé par: models/registry.py
│
└── registry.py
    ├── imports: models/cnn, models/tabnet, models/sklearn_models
    └── utilisé par: Potentiellement phases/* (mais phase3 utilise _build_models() au lieu)
```

### Modules Isolés/Morts

```
ISOLÉS:
├── main_pipeline.py 🔴 CODE MORT
│   ├── classe: IRPPipeline (762 lignes)
│   ├── imports: core, models, evaluation_3d, realtime_visualizer, ui
│   └── utilisé par: AUCUN (sauf verify_irp_compliance.py legacy)
│
├── results_visualizer.py 🟡 CODE MORT PROBABLE
│   ├── classe: ResultsVisualizer (842 lignes)
│   ├── imports: tkinter, matplotlib
│   └── utilisé par: verify_irp_compliance.py (legacy)
│
└── realtime_visualizer.py 🟡 OPTIONNEL
    ├── classe: RealTimeVisualizer (649 lignes)
    ├── imports: tkinter, matplotlib, seaborn
    └── utilisé par: main_pipeline.py (mort) avec try/except
```

---

## 🔍 Détection Cycles - Analyse Approfondie

### Algorithme de Détection (DFS)

```
Graphe d'imports (source → cibles):

app/cli.py → config
app/pipeline_runner.py → config, phases
core/dependencies.py → system_monitor, feature_analyzer, irp_features_requirements
core/dataset_loader.py → core (via dependencies.py)
core/data_harmonization.py → core (via dependencies.py)
evaluation_3d.py → core, evaluation
main_pipeline.py → core, models, evaluation_3d, realtime_visualizer, ui
phases/phase1_config_search.py → config, core
phases/phase2_apply_best_config.py → core
phases/phase3_evaluation.py → core, evaluation_3d, models
ui/__init__.py → ui
```

### Cycles Détectés

**Aucun cycle direct détecté** ✅

**Couplage indirect via `dependencies.py`:**
```
core/dependencies.py → system_monitor.py
core/dependencies.py → feature_analyzer.py
core/dependencies.py → irp_features_requirements.py
    ↓
core/dataset_loader.py → dependencies.py (imports SystemMonitor, FeatureAnalyzer)
core/data_harmonization.py → dependencies.py (imports FeatureAnalyzer, IRPFeaturesRequirements)
    ↓
core/__init__.py → dataset_loader.py, data_harmonization.py
    ↓
phases/* → core (via __init__.py)
```

**Impact:**
- Pas de cycle (pas de retour vers dependencies.py)
- Mais couplage fort via god module

---

## 📦 Redondances Détaillées

### 1. Transform Test Duplication

**Code dupliqué:**
- `preprocessing_pipeline.py:738` → `transform_test()` (58 lignes)
- `phase3_evaluation.py:373` → `_transform_test_fold()` (31 lignes)

**Lignes similaires:**
```python
# Les deux font:
# 1. Numeric coercion + inf to NaN
# 2. Impute using TRAIN-fitted imputer
# 3. Feature selection (if fitted)
# 4. Scaling (if fitted)
```

**Différences:**
- `transform_test()`: Utilise `sanitize_numeric_values()`, plus robuste
- `_transform_test_fold()`: Logique manuelle, moins robuste

**Solution:**
- Supprimer `_transform_test_fold()`, utiliser `transform_test()`

### 2. Visualisation Duplication

**Modules:** 3 modules avec logique matplotlib similaire

**Fonctions similaires:**
- Bar charts (présents dans les 3)
- Scatter plots (présents dans les 3)
- Heatmaps (présents dans 2)
- Sauvegarde de figures (3 implémentations différentes)

**Solution:**
- Extraire helpers communs dans `src/utils/viz_helpers.py`

### 3. Model Building Duplication

**Fichiers:**
- `phase3_evaluation.py:405` → `_build_models()`
- `models/registry.py:21` → `get_model_registry()`

**Logique similaire:**
- Construction de modèles avec try/except
- Gestion des options (CNN, TabNet optionnels)

**Solution:**
- Phase 3 devrait utiliser `get_model_registry(config)`

### 4. Path/File Utilities Duplication

**Pattern répété:** `Path(...).mkdir(parents=True, exist_ok=True)`

**Occurrences:** ~20+ fois dans le codebase

**Solution:**
- Créer `ensure_dir()` helper

### 5. Try/Except Imports Duplication

**Pattern répété:**
```python
try:
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
```

**Occurrences:** ~10 fois (matplotlib, shap, lime, torch)

**Solution:**
- Créer `optional_import()` helper

---

## 🗂️ Fichiers par Statut

### Actifs et Utilisés ✅

- `app/pipeline_runner.py` - Orchestrateur principal
- `app/cli.py` - Interface CLI
- `config.py` - Configuration centralisée
- `phases/phase*.py` - Les 5 phases
- `core/__init__.py`, `core/preprocessing_pipeline.py`, `core/dataset_loader.py`, `core/data_harmonization.py`, `core/feature_engineering.py`, `core/model_utils.py`
- `evaluation/visualizations.py` - Visualisations Phase 3
- `evaluation_3d.py` - Framework évaluation
- `models/cnn.py`, `models/tabnet.py`, `models/sklearn_models.py`, `models/registry.py`
- `ahp_topsis_framework.py` - Framework AHP-TOPSIS

### Code Mort 🔴

- `main_pipeline.py` - IRPPipeline jamais utilisé
- `results_visualizer.py` - Seulement verify_irp_compliance.py (legacy)

### Optionnels 🟡

- `realtime_visualizer.py` - GUI optionnel, peu utilisé
- `system_monitor.py` - Monitoring optionnel
- `ui/features_popup.py` - GUI optionnel

### God Modules ⚠️

- `core/dependencies.py` - Centralise trop d'exports

---

## 📊 Recommandations Finales

### Immédiat (Safe, High Impact)

1. ✅ **Factoriser `transform_test()`** - Phase 3 utilise `transform_test()` directement
2. ✅ **Supprimer `main_pipeline.py`** - Code mort confirmé
3. ✅ **Découpler `dependencies.py`** - Créer utils/ modules

### Court Terme (Medium Impact)

4. ✅ **Consolider visualisations** - Helpers communs
5. ✅ **Utiliser `get_model_registry()`** - Phase 3 utilise registry
6. ✅ **Simplifier DatasetLoader** - Méthodes privées

### Moyen Terme (Low Impact)

7. ✅ **Extraire helpers optionnels** - optional_import()
8. ✅ **Extraire path utilities** - ensure_dir()

### Long Terme (Architecture)

9. ⚠️ **Refactorer visualisations** - Unifier 3 modules
10. ⚠️ **Typing cohérent** - Ajouter type hints partout

---

## 📈 Impact Estimé

### Réduction de Code

| Action | Lignes Supprimées | Maintenance |
|--------|-------------------|-------------|
| Supprimer `main_pipeline.py` | -762 | ⬇️ Forte |
| Supprimer `results_visualizer.py` | -842 | ⬇️ Forte |
| Factoriser `transform_test()` | -31 | ⬇️ Moyenne |
| Consolider visualisations | -100 (estimé) | ⬇️ Moyenne |
| **TOTAL** | **~-1,735 lignes** | |

### Amélioration Architecture

| Action | Couplage | Testabilité | Maintenabilité |
|--------|----------|-------------|----------------|
| Découpler `dependencies.py` | ⬇️ Forte | ⬆️ Forte | ⬆️ Forte |
| Factoriser visualisations | ⬇️ Moyenne | ⬆️ Moyenne | ⬆️ Forte |
| Simplifier DatasetLoader | ⬇️ Faible | ⬆️ Moyenne | ⬆️ Moyenne |

---

**Rapport généré par:** Analyse Statique Manuelle  
**Date:** 2024-01-18  
**Version:** 1.0
