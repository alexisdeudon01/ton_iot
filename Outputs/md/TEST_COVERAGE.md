# Test Coverage Documentation

**Last Updated**: 2024-01-18

## 📋 Vue d'ensemble

Le fichier `main_test.py` est un runner de tests unifié qui exécute tous les tests pytest du projet avec:
- **Logs détaillés** : input/output à chaque étape
- **Explications** : pourquoi chaque test réussit/échoue
- **Diagrammes** : générés dans `output/test_reports/test_run_TIMESTAMP/`
- **Rapports** : Markdown détaillé et JSON avec tous les résultats

## 🎯 Couverture des Tests

### 1. Tests de Preprocessing Pipeline

**Fichiers testés** : `tests/test_preprocessing_pipeline.py`

**Couverture** :
- ✅ Preprocessing stateless (nettoyage, encodage)
- ✅ Transform methods (`transform_test()`, `transform_data()`)
- ✅ Zéro data leakage (scaler/selector fitté uniquement sur TRAIN)
- ✅ Sanitization numérique (inf, outliers)
- ✅ Validation pipeline fitted state

**Exemples de tests** :
- `test_sanitize_numeric_values_removes_inf_and_clips`
- `test_transform_test_requires_fitted_pipeline`
- `test_no_data_leakage_scaler_fit_only_on_train`

---

### 2. Tests Phase 2 (Apply Best Configuration)

**Fichiers testés** : `tests/test_phase2_outputs.py`

**Couverture** :
- ✅ Génération de `best_preprocessed.parquet` (ou `.csv.gz`)
- ✅ Génération de `feature_names.json`
- ✅ Génération de `phase2_summary.md`
- ✅ Présence et encodage correct de `dataset_source` (0/1)
- ✅ Preprocessing stateless uniquement (pas de scaling/FS/SMOTE)

**Input** : Mock config et dataset avec `dataset_source`

**Expected Output** :
- Fichier preprocessed avec `dataset_source` encodé
- JSON avec liste des features
- Summary Markdown avec statistiques

---

### 3. Tests Phase 3 (3D Evaluation)

**Fichiers testés** : 
- `tests/test_phase3_synthetic.py`
- `tests/test_phase3_cnn_tabnet.py`

**Couverture** :
- ✅ Mode synthétique (`--synthetic` flag)
- ✅ Preprocessing model-aware par fold
- ✅ Génération de CSV (evaluation_results.csv, dimension_scores.csv)
- ✅ Génération de rapports algorithmes (algorithm_reports/*.md)
- ✅ Génération de visualisations (visualizations/*.png)
- ✅ Support CNN/TabNet avec reshape et class_weight

**Input** : Config avec `synthetic_mode=True` ou datasets réels

**Expected Output** :
- CSV avec métriques (F1, accuracy, precision, recall, temps, mémoire, explainability)
- Rapports Markdown par algorithme
- Diagrammes de visualisation
- INDEX.md pour métriques et visualisations

---

### 4. Tests Model-Aware Preprocessing Profiles

**Fichiers testés** : `tests/test_model_aware_profiles.py`

**Couverture** :
- ✅ Profil LR : scaling=True, feature_selection=True, resampling=True
- ✅ Profil Tree (DT/RF) : scaling=False, feature_selection=False, class_weight='balanced'
- ✅ Profil CNN : scaling=True, feature_selection=False, resampling=True, cnn_reshape=True
- ✅ Profil TabNet : scaling=False, feature_selection=False, class_weight='balanced'
- ✅ Calcul dynamique de `feature_selection_k`

**Input** : Noms de modèles (LR, DT, RF, CNN, TabNet)

**Expected Output** : Profiles avec bonnes valeurs booléennes et paramètres

---

### 5. Tests Zéro Data Leakage

**Fichiers testés** : `tests/test_no_data_leakage.py`

**Couverture** :
- ✅ Scaler fitté uniquement sur TRAIN fold
- ✅ Test transformé avec scaler fitté sur TRAIN
- ✅ Feature selector fitté uniquement sur TRAIN
- ✅ Test transformé avec selector fitté sur TRAIN
- ✅ Imputer fitté uniquement sur TRAIN (médiane calculée sur TRAIN)

**Input** : TRAIN et TEST DataFrames avec distributions différentes

**Expected Output** : TEST transformé avec statistiques (moyenne, médiane) basées sur TRAIN uniquement

---

### 6. Tests Algorithm Handling

**Fichiers testés** : `tests/test_algo_handling.py`

**Couverture** :
- ✅ `get_algo_names()` : extraction des noms d'algorithmes depuis DataFrame
- ✅ `ensure_algo_column()` : création de colonne 'algo' si manquante
- ✅ `sanitize_algo_name()` : nettoyage des noms pour fichiers (espaces → underscore)

**Input** : DataFrames avec colonne 'algo' ou index nommé 'algo'

**Expected Output** : Series avec noms d'algorithmes normalisés

---

### 7. Tests Dataset Source Flag

**Fichiers testés** : `tests/test_dataset_source_flag.py`

**Couverture** :
- ✅ Flag `phase3_use_dataset_source` contrôle inclusion de `dataset_source`
- ✅ `dataset_source` préservé quand flag=True
- ✅ `dataset_source` exclu quand flag=False

**Input** : Config avec flag True/False, DataFrame avec `dataset_source`

**Expected Output** : Features avec/sans `dataset_source` selon flag

---

### 8. Tests CNN/TabNet

**Fichiers testés** : 
- `tests/test_cnn.py`
- `tests/test_tabnet.py`

**Couverture** :
- ✅ Initialisation CNN/TabNet
- ✅ Training sur données synthétiques
- ✅ Prediction et predict_proba
- ✅ Validation sklearn interface (fit/predict/predict_proba)
- ✅ Reshape CNN pour input (n, d, 1)
- ✅ Class_weight pour TabNet

**Input** : Données synthétiques (X, y) binaires/multiclass

**Expected Output** : Modèle entraîné avec prédictions valides

---

## 📊 Structure des Rapports Générés

### Répertoire de sortie
```
output/test_reports/test_run_YYYYMMDD_HHMMSS/
├── test_coverage_diagram.png    # Diagramme de couverture par catégorie
├── test_report.md               # Rapport Markdown détaillé
└── test_results.json            # Résultats JSON complets
```

### Contenu du rapport Markdown

1. **Summary Statistics** :
   - Total de tests
   - Passed/Failed/Skipped avec pourcentages

2. **Test Results by Outcome** :
   - ✅ **Passed Tests** : nom, durée, input, output, raison du succès
   - ❌ **Failed Tests** : nom, durée, input, raison de l'échec, message d'erreur, traceback
   - ⏭️ **Skipped Tests** : nom, raison du skip

### Contenu du JSON

```json
{
  "timestamp": "2024-01-18T10:30:00",
  "total_tests": 29,
  "passed": 25,
  "failed": 2,
  "skipped": 2,
  "results": [
    {
      "test_name": "tests/test_phase2_outputs.py::test_phase2_outputs",
      "outcome": "passed",
      "duration": 1.234,
      "input_description": "Mock config and dataset...",
      "output_description": "Preprocessed data file...",
      "success_reason": "Phase 2 outputs successfully generated...",
      ...
    }
  ]
}
```

## 🎨 Diagrammes Générés

### Test Coverage Diagram

1. **Graphique en barres** (gauche) :
   - Nombre de tests Passed/Failed/Skipped par catégorie :
     - Preprocessing
     - Phase 2
     - Phase 3
     - Models (CNN/TabNet/sklearn)
     - Data Leakage
     - Algorithm Handling
     - Dataset Source
     - Model Profiles
     - Other

2. **Graphique en camembert** (droite) :
   - Distribution globale : Passed vs Failed vs Skipped
   - Pourcentages et totaux

## 🔍 Explications de Succès/Échec

### Raisons de succès automatiques

Le plugin génère automatiquement des explications selon le type de test:

- **Preprocessing tests** : "Preprocessing pipeline correctly applied stateless transformations and maintained data integrity"
- **Phase 2 tests** : "Phase 2 outputs successfully generated with correct format"
- **Phase 3 tests** : "Phase 3 evaluation completed with model-aware preprocessing per fold, ensuring zero data leakage"
- **CNN tests** : "CNN model correctly initialized, trained, and evaluated with proper input reshaping"
- **Leakage tests** : "No data leakage detected: scaler/selector fitted only on TRAIN, test transformed using TRAIN-fitted objects"

### Raisons d'échec automatiques

- **AssertionError** : "Assertion failed: [message]"
- **ValueError** : "Invalid value: [message]"
- **AttributeError** : "Missing attribute: [message]"
- **KeyError** : "Missing key: [message]"
- **ImportError** : "Missing dependency: [module]"
- **FileNotFoundError** : "File not found: [path]"

## 📝 Utilisation

### Lancer tous les tests
```bash
python main_test.py
```

### Lancer avec verbose
```bash
python main_test.py -v
```

### Lancer un test spécifique
```bash
pytest tests/test_phase2_outputs.py::test_phase2_outputs -v
```

## 📈 Statistiques Typiques

- **Total de tests** : ~29 fichiers de tests
- **Tests actifs** : ~20-25 (selon dépendances)
- **Temps d'exécution** : 30-120 secondes selon environnement
- **Couverture estimée** :
  - Preprocessing : ~100%
  - Phase 2 : ~100%
  - Phase 3 : ~90% (CNN/TabNet optionnels)
  - Models : ~85% (selon dépendances)

## 🔧 Dépendances pour Tests Complets

**Requis** :
- pytest
- pandas
- numpy
- scikit-learn

**Optionnels** (pour certains tests) :
- torch (CNN tests)
- pytorch-tabnet (TabNet tests)
- shap (explainability tests)
- lime (explainability tests)
- matplotlib (diagram generation)
