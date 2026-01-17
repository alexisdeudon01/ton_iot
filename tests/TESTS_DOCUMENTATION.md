# Documentation Complète des Tests

**Date:** $(date)  
**Branche:** dev2

## Vue d'Ensemble

Ce document détaille chaque test avec:
- **Input**: Ce que le test prend en entrée
- **Traitement**: Ce que le test fait sur cet input
- **Output**: Ce que le test produit/vérifie
- **Méthode**: La méthode/fonction testée

---

## 1. Tests pour `model_utils.py` (`test_model_utils.py`)

### Structure des Imports
```python
import sys
from pathlib import Path
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Ajout du projet root au path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.core.model_utils import fresh_model
```

**Modules testés:** `src.core.model_utils.fresh_model()`

---

### Test 1: `test_fresh_model_sklearn_lr`

| Attribut | Détail |
|----------|--------|
| **Input** | `LogisticRegression(random_state=42, max_iter=100)` - Instance de modèle sklearn non-fittée |
| **Traitement** | Appelle `fresh_model(original)` pour créer une nouvelle instance |
| **Output** | Vérifie que:<br>- `fresh is not original` (instances différentes)<br>- `isinstance(fresh, LogisticRegression)` (même type)<br>- `fresh.random_state == 42` (paramètres préservés)<br>- `fresh.max_iter == 100` (paramètres préservés)<br>- `not hasattr(fresh, 'classes_')` (modèle non-fitté) |
| **Méthode** | `fresh_model()` - utilise `sklearn.base.clone()` |
| **Objectif** | Valider que `fresh_model()` crée une nouvelle instance non-fittée avec paramètres préservés pour LogisticRegression |

---

### Test 2: `test_fresh_model_sklearn_dt`

| Attribut | Détail |
|----------|--------|
| **Input** | `DecisionTreeClassifier(random_state=42, max_depth=5)` - Instance DecisionTree non-fittée |
| **Traitement** | Appelle `fresh_model(original)` |
| **Output** | Vérifie:<br>- Instances différentes<br>- Type correct (`DecisionTreeClassifier`)<br>- `random_state == 42`, `max_depth == 5`<br>- `not hasattr(fresh, 'tree_')` (non-fitté) |
| **Méthode** | `fresh_model()` - utilise `sklearn.base.clone()` |
| **Objectif** | Valider clonage de DecisionTree |

---

### Test 3: `test_fresh_model_sklearn_rf`

| Attribut | Détail |
|----------|--------|
| **Input** | `RandomForestClassifier(random_state=42, n_estimators=50)` - Instance RandomForest non-fittée |
| **Traitement** | Appelle `fresh_model(original)` |
| **Output** | Vérifie:<br>- Instances différentes<br>- Type correct (`RandomForestClassifier`)<br>- Paramètres préservés (`random_state`, `n_estimators`)<br>- `not hasattr(fresh, 'estimators_')` (non-fitté) |
| **Méthode** | `fresh_model()` - utilise `sklearn.base.clone()` |
| **Objectif** | Valider clonage de RandomForest |

---

### Test 4: `test_fresh_model_preserves_parameters`

| Attribut | Détail |
|----------|--------|
| **Input** | `LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', C=0.1)` - Multiple paramètres |
| **Traitement** | Appelle `fresh_model(original)` |
| **Output** | Vérifie que TOUS les paramètres sont préservés:<br>- `random_state == 42`<br>- `max_iter == 1000`<br>- `class_weight == 'balanced'`<br>- `C == 0.1` |
| **Méthode** | `fresh_model()` - clonage avec préservation complète des paramètres |
| **Objectif** | Valider que tous les paramètres sont préservés, pas seulement quelques-uns |

---

### Test 5: `test_fresh_model_unfitted_after_fitting`

| Attribut | Détail |
|----------|--------|
| **Input** | `LogisticRegression(random_state=42)` + données `X` (100, 10) et `y` (100,) - Modèle FIT puis cloné |
| **Traitement** | 1. Fit `original.fit(X, y)`<br>2. Vérifie `original` est fitté<br>3. Appelle `fresh_model(original)` |
| **Output** | Vérifie:<br>- `fresh is not original`<br>- `original` a toujours `classes_` (fitté)<br>- `fresh` n'a PAS `classes_` (non-fitté)<br>- `isinstance(fresh, LogisticRegression)` |
| **Méthode** | `fresh_model()` - doit créer instance non-fittée même depuis modèle fitté |
| **Objectif** | Valider que `fresh_model()` nettoie l'état fitté (critical pour éviter data leakage) |

---

### Test 6: `test_fresh_model_cnn`

| Attribut | Détail |
|----------|--------|
| **Input** | `CNNTabularClassifier(hidden_dims=[64, 32], learning_rate=0.001, batch_size=32, random_state=42)` - Modèle CNN personnalisé |
| **Traitement** | Try/except pour importer CNN, sinon skip. Appelle `fresh_model(original)` |
| **Output** | Vérifie:<br>- Instances différentes<br>- Type correct (`CNNTabularClassifier`)<br>- Paramètres préservés (`learning_rate`, `batch_size`, `random_state`, `hidden_dims`)<br>- `fresh.model is None` (modèle PyTorch non initialisé) |
| **Méthode** | `fresh_model()` - utilise `get_params()` ou `deepcopy()` pour modèles non-sklearn |
| **Objectif** | Valider clonage de modèles personnalisés (CNN) |

---

### Test 7: `test_fresh_model_tabnet`

| Attribut | Détail |
|----------|--------|
| **Input** | `TabNetClassifierWrapper(n_d=8, n_a=8, seed=42, max_epochs=50)` - Modèle TabNet |
| **Traitement** | Try/except pour importer TabNet, sinon skip. Appelle `fresh_model(original)` |
| **Output** | Vérifie:<br>- Instances différentes<br>- Type correct (`TabNetClassifierWrapper`)<br>- Paramètres préservés (`n_d`, `n_a`, `seed`, `max_epochs`)<br>- `fresh.model is None` (modèle non initialisé) |
| **Méthode** | `fresh_model()` - utilise `get_params()` ou `deepcopy()` |
| **Objectif** | Valider clonage de TabNet |

---

### Test 8: `test_fresh_model_independence`

| Attribut | Détail |
|----------|--------|
| **Input** | `LogisticRegression(random_state=42)` + données `X` (100, 10), `y` (100,) |
| **Traitement** | 1. Crée `original` et `fresh = fresh_model(original)`<br>2. Fit seulement `fresh.fit(X, y)` |
| **Output** | Vérifie:<br>- `original` reste non-fitté (`not hasattr(original, 'classes_')`)<br>- `fresh` devient fitté (`hasattr(fresh, 'classes_')`) |
| **Méthode** | `fresh_model()` - indépendance complète des instances |
| **Objectif** | Valider que les instances sont indépendantes (pas de side effects) |

---

## 2. Tests pour `cnn.py` (`test_cnn.py`)

### Structure des Imports
```python
import sys
from pathlib import Path
import pytest
import numpy as np

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

pytest.importorskip("torch", reason="torch not available")

from src.models.cnn import TabularCNN, CNNTabularClassifier, TabularDataset
import torch
```

**Modules testés:** `TabularDataset`, `TabularCNN`, `CNNTabularClassifier`

---

### Test 1: `test_tabular_dataset`

| Attribut | Détail |
|----------|--------|
| **Input** | `X`: numpy array (100, 10) float32, `y`: numpy array (100,) int64 |
| **Traitement** | 1. Crée `TabularDataset(X, y)`<br>2. Vérifie `len(dataset)`<br>3. Indexe `dataset[0]`<br>4. Crée `TabularDataset(X, None)` (sans labels) |
| **Output** | Vérifie:<br>- `len(dataset) == 100`<br>- `dataset[0]` retourne tuple `(x_item, y_item)`<br>- `x_item.shape == (10,)`<br>- `y_item.item() in [0, 1]`<br>- Dataset sans labels fonctionne (`x_item_only.shape == (10,)`) |
| **Méthode** | `TabularDataset.__init__()`, `__len__()`, `__getitem__()` |
| **Objectif** | Valider création et indexation du Dataset PyTorch |

---

### Test 2: `test_tabular_cnn_init_default`

| Attribut | Détail |
|----------|--------|
| **Input** | `TabularCNN(input_dim=10, num_classes=2)` - Paramètres minimaux (hidden_dims=None) |
| **Traitement** | Crée instance avec paramètres par défaut |
| **Output** | Vérifie:<br>- `model.input_dim == 10`<br>- `model.num_classes == 2`<br>- `hasattr(model, 'conv_layers')`<br>- `hasattr(model, 'fc_layers')` |
| **Méthode** | `TabularCNN.__init__()` - utilise `hidden_dims=[64, 32, 16]` par défaut |
| **Objectif** | Valider initialisation avec valeurs par défaut |

---

### Test 3: `test_tabular_cnn_init_custom`

| Attribut | Détail |
|----------|--------|
| **Input** | `TabularCNN(input_dim=20, num_classes=3, hidden_dims=[64, 32])` - Paramètres personnalisés |
| **Traitement** | Crée instance avec `hidden_dims` personnalisés |
| **Output** | Vérifie:<br>- `model.input_dim == 20`<br>- `model.num_classes == 3`<br>- `len(model.conv_layers) > 0` (layers créés) |
| **Méthode** | `TabularCNN.__init__()` - crée conv layers selon `hidden_dims` |
| **Objectif** | Valider initialisation avec architecture personnalisée |

---

### Test 4: `test_tabular_cnn_init_empty_hidden_dims`

| Attribut | Détail |
|----------|--------|
| **Input** | `TabularCNN(input_dim=10, num_classes=2, hidden_dims=[])` - `hidden_dims` vide |
| **Traitement** | Tente de créer instance avec `hidden_dims=[]` |
| **Output** | **Vérifie qu'une `ValueError` est levée** avec message "hidden_dims cannot be empty" |
| **Méthode** | `TabularCNN.__init__()` - validation des paramètres |
| **Objectif** | Valider que le modèle rejette les configurations invalides |

---

### Test 5: `test_tabular_cnn_forward`

| Attribut | Détail |
|----------|--------|
| **Input** | `TabularCNN(input_dim=10, num_classes=2, hidden_dims=[32, 16])` + tensor `x` (8, 10) |
| **Traitement** | 1. `model.eval()`<br>2. Forward pass `model(x)` avec `torch.no_grad()` |
| **Output** | Vérifie:<br>- `output.shape == (8, 2)` (batch_size, num_classes)<br>- `not torch.isnan(output).any()` (pas de NaN) |
| **Méthode** | `TabularCNN.forward()` - propagation avant |
| **Objectif** | Valider que forward pass produit sortie correcte sans NaN |

---

### Test 6: `test_tabular_cnn_forward_small_input`

| Attribut | Détail |
|----------|--------|
| **Input** | `TabularCNN(input_dim=5, num_classes=2, hidden_dims=[16])` + tensor `x` (4, 5) - Petite dimension |
| **Traitement** | Forward pass avec petite dimension d'entrée |
| **Output** | Vérifie:<br>- `output.shape == (4, 2)`<br>- `not torch.isnan(output).any()` |
| **Méthode** | `TabularCNN.forward()` - edge case avec petite dimension |
| **Objectif** | Valider que le modèle fonctionne avec petites dimensions (test edge case) |

---

### Test 7: `test_cnn_classifier_init`

| Attribut | Détail |
|----------|--------|
| **Input** | `CNNTabularClassifier(hidden_dims=[64, 32], learning_rate=0.001, batch_size=32, epochs=5, random_state=42)` |
| **Traitement** | Crée instance du classifier sklearn-compatible |
| **Output** | Vérifie:<br>- Tous les paramètres sont assignés correctement<br>- `clf.model is None` (modèle PyTorch non initialisé) |
| **Méthode** | `CNNTabularClassifier.__init__()` |
| **Objectif** | Valider initialisation du wrapper sklearn |

---

### Test 8: `test_cnn_classifier_init_empty_hidden_dims`

| Attribut | Détail |
|----------|--------|
| **Input** | `CNNTabularClassifier(hidden_dims=[])` - `hidden_dims` vide |
| **Traitement** | Tente de créer instance avec `hidden_dims=[]` |
| **Output** | **Vérifie qu'une `ValueError` est levée** avec message "hidden_dims cannot be empty" |
| **Méthode** | `CNNTabularClassifier.__init__()` - validation dans wrapper |
| **Objectif** | Valider validation des paramètres au niveau wrapper (pas seulement dans TabularCNN) |

---

### Test 9: `test_cnn_classifier_fit_predict`

| Attribut | Détail |
|----------|--------|
| **Input** | `CNNTabularClassifier(...)` + données synthétiques `X` (200, 10), `y` (200,) |
| **Traitement** | 1. `clf.fit(X, y)` - entraîne le modèle<br>2. `clf.predict(X[:10])` - prédictions |
| **Output** | Vérifie:<br>- `clf.model is not None` (modèle créé)<br>- `clf.input_dim == 10`<br>- `hasattr(clf.label_encoder, 'classes_')` (labels encodés)<br>- `len(y_pred) == 10`<br>- `all(pred in clf.label_encoder.classes_ for pred in y_pred)` |
| **Méthode** | `CNNTabularClassifier.fit()`, `predict()` |
| **Objectif** | Valider cycle complet fit/predict (end-to-end) |

---

### Test 10: `test_cnn_classifier_predict_proba`

| Attribut | Détail |
|----------|--------|
| **Input** | `CNNTabularClassifier(...)` + données `X` (200, 10), `y` (200,) |
| **Traitement** | 1. `clf.fit(X, y)`<br>2. `clf.predict_proba(X[:10])` |
| **Output** | Vérifie:<br>- `proba.shape == (10, 2)` (samples, classes)<br>- `np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)` (probabilités somment à 1)<br>- `(proba >= 0).all() and (proba <= 1).all()` (valeurs dans [0, 1]) |
| **Méthode** | `CNNTabularClassifier.predict_proba()` |
| **Objectif** | Valider que probabilités sont valides (somme=1, valeurs dans [0,1]) |

---

### Test 11: `test_cnn_classifier_multiclass`

| Attribut | Détail |
|----------|--------|
| **Input** | `CNNTabularClassifier(...)` + données `X` (300, 10), `y` (300,) avec 3 classes (0, 1, 2) |
| **Traitement** | 1. `clf.fit(X, y)`<br>2. `clf.predict()` et `clf.predict_proba()` |
| **Output** | Vérifie:<br>- `len(np.unique(y)) == 3` (3 classes)<br>- `proba.shape == (10, 3)` (3 classes)<br>- `np.allclose(proba.sum(axis=1), 1.0)` (probabilités valides) |
| **Méthode** | `CNNTabularClassifier.fit()`, `predict()`, `predict_proba()` - multiclass |
| **Objectif** | Valider que le modèle gère la classification multiclasse |

---

### Test 12: `test_cnn_classifier_sklearn_interface`

| Attribut | Détail |
|----------|--------|
| **Input** | `CNNTabularClassifier(...)` - Instance du classifier |
| **Traitement** | Vérifie présence des méthodes sklearn standard |
| **Output** | Vérifie `hasattr()` pour:<br>- `fit`<br>- `predict`<br>- `predict_proba`<br>- `get_params`<br>- `set_params` |
| **Méthode** | Vérification d'interface (pas d'appel réel) |
| **Objectif** | Valider que le classifier suit l'interface sklearn (compatibilité) |

---

## 3. Tests pour `tabnet.py` (`test_tabnet.py`)

### Structure des Imports
```python
import sys
from pathlib import Path
import pytest
import numpy as np

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

pytest.importorskip("pytorch_tabnet", reason="pytorch-tabnet not available")

from src.models.tabnet import TabNetClassifierWrapper
```

**Modules testés:** `TabNetClassifierWrapper`

---

### Test 1: `test_tabnet_classifier_init_default`

| Attribut | Détail |
|----------|--------|
| **Input** | `TabNetClassifierWrapper()` - Initialisation avec paramètres par défaut |
| **Traitement** | Crée instance sans arguments |
| **Output** | Vérifie valeurs par défaut:<br>- `clf.n_d == 8`<br>- `clf.n_a == 8`<br>- `clf.n_steps == 3`<br>- `clf.seed == 42`<br>- `clf.max_epochs == 100`<br>- `clf.model is None` (non initialisé)<br>- `clf.input_dim is None` |
| **Méthode** | `TabNetClassifierWrapper.__init__()` |
| **Objectif** | Valider valeurs par défaut du wrapper |

---

### Test 2: `test_tabnet_classifier_init_custom`

| Attribut | Détail |
|----------|--------|
| **Input** | `TabNetClassifierWrapper(n_d=16, n_a=16, n_steps=5, gamma=2.0, seed=123, max_epochs=50)` |
| **Traitement** | Crée instance avec paramètres personnalisés |
| **Output** | Vérifie que tous les paramètres sont assignés correctement |
| **Méthode** | `TabNetClassifierWrapper.__init__()` |
| **Objectif** | Valider assignation de paramètres personnalisés |

---

### Test 3: `test_tabnet_classifier_fit_predict`

| Attribut | Détail |
|----------|--------|
| **Input** | `TabNetClassifierWrapper(...)` + données `X` (300, 10), `y` (300,) |
| **Traitement** | 1. `clf.fit(X, y)` avec `max_epochs=5` (court)<br>2. `clf.predict(X[:20])` |
| **Output** | Vérifie:<br>- `clf.model is not None`<br>- `clf.input_dim == 10`<br>- `hasattr(clf.label_encoder, 'classes_')`<br>- `len(y_pred) == 20`<br>- Prédictions dans classes valides |
| **Méthode** | `TabNetClassifierWrapper.fit()`, `predict()` |
| **Objectif** | Valider cycle complet fit/predict pour TabNet |

---

### Test 4: `test_tabnet_classifier_predict_proba`

| Attribut | Détail |
|----------|--------|
| **Input** | `TabNetClassifierWrapper(...)` + données `X` (300, 10), `y` (300,) |
| **Traitement** | 1. `clf.fit(X, y)`<br>2. `clf.predict_proba(X[:20])` |
| **Output** | Vérifie:<br>- `proba.shape == (20, 2)`<br>- `np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)`<br>- `(proba >= 0).all() and (proba <= 1).all()` |
| **Méthode** | `TabNetClassifierWrapper.predict_proba()` |
| **Objectif** | Valider probabilités valides pour TabNet |

---

### Test 5: `test_tabnet_classifier_multiclass`

| Attribut | Détail |
|----------|--------|
| **Input** | `TabNetClassifierWrapper(...)` + données `X` (400, 10), `y` (400,) avec 3 classes |
| **Traitement** | 1. `clf.fit(X, y)`<br>2. `clf.predict()` et `clf.predict_proba()` |
| **Output** | Vérifie:<br>- `len(np.unique(y)) == 3`<br>- `proba.shape == (20, 3)`<br>- `np.allclose(proba.sum(axis=1), 1.0)` |
| **Méthode** | `TabNetClassifierWrapper.fit()`, `predict()`, `predict_proba()` - multiclass |
| **Objectif** | Valider classification multiclasse pour TabNet |

---

### Test 6: `test_tabnet_classifier_sklearn_interface`

| Attribut | Détail |
|----------|--------|
| **Input** | `TabNetClassifierWrapper(...)` |
| **Traitement** | Vérifie présence des méthodes sklearn |
| **Output** | Vérifie `hasattr()` pour `fit`, `predict`, `predict_proba`, `get_params`, `set_params` |
| **Méthode** | Vérification d'interface |
| **Objectif** | Valider interface sklearn pour TabNet |

---

### Test 7: `test_tabnet_classifier_parameters_preserved`

| Attribut | Détail |
|----------|--------|
| **Input** | `TabNetClassifierWrapper(n_d=16, n_a=16, gamma=2.0, seed=99, max_epochs=10)` + données |
| **Traitement** | 1. Crée avec paramètres<br>2. `clf.fit(X, y)`<br>3. Vérifie paramètres après fit |
| **Output** | Vérifie que paramètres sont préservés après fit:<br>- `clf.n_d == 16`<br>- `clf.n_a == 16`<br>- `clf.gamma == 2.0`<br>- `clf.seed == 99` |
| **Méthode** | `TabNetClassifierWrapper.fit()` - doit préserver paramètres init |
| **Objectif** | Valider que fit() ne modifie pas les paramètres init |

---

## 4. Tests pour `explainability.py` (`test_explainability.py`)

### Structure des Imports
```python
import sys
from pathlib import Path
import pytest
import numpy as np

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.evaluation.explainability import (
    get_native_interpretability_score,
    compute_shap_score,
    compute_lime_score,
    compute_explainability_score
)

pytest.importorskip("shap", reason="shap not available")
pytest.importorskip("lime", reason="lime not available")
```

**Modules testés:** `explainability.py` (SHAP, LIME, native scores)

---

### Test 1: `test_get_native_interpretability_score`

| Attribut | Détail |
|----------|--------|
| **Input** | Noms de modèles: `'Logistic_Regression'`, `'Decision_Tree'`, `'Random_Forest'`, `'CNN'`, `'TabNet'`, `'UnknownModel'` |
| **Traitement** | Appelle `get_native_interpretability_score(model_name)` pour chaque |
| **Output** | Vérifie scores:<br>- `Logistic_Regression == 1.0` (hautement interprétable)<br>- `Decision_Tree == 1.0` (hautement interprétable)<br>- `Random_Forest == 1.0` (hautement interprétable)<br>- `CNN == 0.0` (black box)<br>- `TabNet == 0.0` (black box)<br>- `UnknownModel == 0.5` (valeur par défaut) |
| **Méthode** | `get_native_interpretability_score()` - lookup dans dictionnaire |
| **Objectif** | Valider scoring natif pour différents types de modèles |

---

### Test 2: `test_compute_shap_score_tree_model`

| Attribut | Détail |
|----------|--------|
| **Input** | `RandomForestClassifier` fitté + `X_sample` (50, 10), `top_k=5` |
| **Traitement** | 1. Crée et fit `RandomForestClassifier(n_estimators=10)`<br>2. Appelle `compute_shap_score(model, X_sample, top_k=5)` |
| **Output** | Vérifie:<br>- `shap_score is not None`<br>- `isinstance(shap_score, float)`<br>- `shap_score >= 0` |
| **Méthode** | `compute_shap_score()` - utilise `TreeExplainer` pour arbres |
| **Objectif** | Valider calcul SHAP pour modèles tree-based |

---

### Test 3: `test_compute_shap_score_linear_model`

| Attribut | Détail |
|----------|--------|
| **Input** | `LogisticRegression` fitté + `X_sample` (50, 10), `top_k=5` |
| **Traitement** | 1. Crée et fit `LogisticRegression(max_iter=1000)`<br>2. Appelle `compute_shap_score(model, X_sample, top_k=5)` |
| **Output** | Vérifie:<br>- `shap_score is not None`<br>- `isinstance(shap_score, float)`<br>- `shap_score >= 0` |
| **Méthode** | `compute_shap_score()` - utilise `KernelExplainer` pour modèles non-tree |
| **Objectif** | Valider calcul SHAP pour modèles linéaires (KernelExplainer) |

---

### Test 4: `test_compute_shap_score_failure_handling`

| Attribut | Détail |
|----------|--------|
| **Input** | Classe `BadModel` avec `predict_proba()` qui lève `ValueError` + `X_sample` (10, 5) |
| **Traitement** | Appelle `compute_shap_score(BadModel(), X_sample)` |
| **Output** | Vérifie:<br>- `shap_score is None` (gestion d'erreur) |
| **Méthode** | `compute_shap_score()` - try/except pour gérer erreurs |
| **Objectif** | Valider gestion gracieuse des erreurs (ne pas crasher) |

---

### Test 5: `test_compute_lime_score`

| Attribut | Détail |
|----------|--------|
| **Input** | `RandomForestClassifier` fitté + `X_sample` (50, 10), `y_sample` (50,), `top_k=5` |
| **Traitement** | 1. Crée et fit `RandomForestClassifier`<br>2. Appelle `compute_lime_score(model, X_sample, y_sample, top_k=5)` |
| **Output** | Vérifie:<br>- `lime_score is not None`<br>- `isinstance(lime_score, float)`<br>- `lime_score >= 0` |
| **Méthode** | `compute_lime_score()` - utilise `LimeTabularExplainer` |
| **Objectif** | Valider calcul LIME score |

---

### Test 6: `test_compute_lime_score_failure_handling`

| Attribut | Détail |
|----------|--------|
| **Input** | Classe `BadModel` avec `predict_proba()` qui lève `ValueError` + données |
| **Traitement** | Appelle `compute_lime_score(BadModel(), X_sample, y_sample)` |
| **Output** | Vérifie:<br>- `lime_score is None` |
| **Méthode** | `compute_lime_score()` - try/except |
| **Objectif** | Valider gestion d'erreurs pour LIME |

---

### Test 7: `test_compute_explainability_score_all_components`

| Attribut | Détail |
|----------|--------|
| **Input** | `model_name='Random_Forest'`, `shap_score=0.5`, `lime_score=0.3`, `weights=(0.5, 0.3, 0.2)` |
| **Traitement** | Appelle `compute_explainability_score(...)` avec tous les composants |
| **Output** | Vérifie dict résultat:<br>- Clés présentes: `explain_score`, `native_score`, `shap_score`, `lime_score`, `weights_used`, `missing_components`<br>- `native_score == 1.0`<br>- `shap_score == 0.5`<br>- `lime_score == 0.3`<br>- `missing_components == []`<br>- `0 <= explain_score <= 1` |
| **Méthode** | `compute_explainability_score()` - combinaison pondérée |
| **Objectif** | Valider calcul score d'explainability avec tous les composants |

---

### Test 8: `test_compute_explainability_score_missing_shap`

| Attribut | Détail |
|----------|--------|
| **Input** | `model_name='Logistic_Regression'`, `shap_score=None`, `lime_score=0.4`, `weights=(0.5, 0.3, 0.2)` |
| **Traitement** | Appelle `compute_explainability_score(...)` sans SHAP |
| **Output** | Vérifie:<br>- `'shap' in missing_components`<br>- `np.isnan(result['shap_score'])`<br>- `result['lime_score'] == 0.4`<br>- `sum(weights_used.values()) == 1.0` (renormalisé) |
| **Méthode** | `compute_explainability_score()` - renormalisation des poids |
| **Objectif** | Valider renormalisation des poids quand SHAP manquant |

---

### Test 9: `test_compute_explainability_score_missing_lime`

| Attribut | Détail |
|----------|--------|
| **Input** | `model_name='CNN'`, `shap_score=0.5`, `lime_score=None`, `weights=(0.5, 0.3, 0.2)` |
| **Traitement** | Appelle `compute_explainability_score(...)` sans LIME |
| **Output** | Vérifie:<br>- `'lime' in missing_components`<br>- `np.isnan(result['lime_score'])`<br>- `result['shap_score'] == 0.5`<br>- `result['native_score'] == 0.0` (CNN)<br>- `sum(weights_used.values()) == 1.0` |
| **Méthode** | `compute_explainability_score()` - renormalisation |
| **Objectif** | Valider renormalisation quand LIME manquant |

---

### Test 10: `test_compute_explainability_score_native_only`

| Attribut | Détail |
|----------|--------|
| **Input** | `model_name='Decision_Tree'`, `shap_score=None`, `lime_score=None`, `weights=(0.5, 0.3, 0.2)` |
| **Traitement** | Appelle `compute_explainability_score(...)` avec seulement native |
| **Output** | Vérifie:<br>- `len(missing_components) == 2` (shap et lime)<br>- `native_score == 1.0`<br>- `weights_used['native'] == 1.0` (renormalisé à 100%) |
| **Méthode** | `compute_explainability_score()` - fallback sur native seulement |
| **Objectif** | Valider fallback sur native score si SHAP/LIME absents |

---

### Test 11: `test_compute_explainability_score_nan_handling`

| Attribut | Détail |
|----------|--------|
| **Input** | `model_name='Random_Forest'`, `shap_score=np.nan`, `lime_score=0.3`, `weights=(0.5, 0.3, 0.2)` |
| **Traitement** | Appelle `compute_explainability_score(...)` avec NaN |
| **Output** | Vérifie:<br>- `'shap' in missing_components` (NaN traité comme manquant)<br>- `np.isnan(result['shap_score'])`<br>- `result['lime_score'] == 0.3` |
| **Méthode** | `compute_explainability_score()` - détection de NaN |
| **Objectif** | Valider traitement de NaN (traité comme None) |

---

## Résumé de Cohérence des Tests

### ✅ Points Forts

1. **Structure uniforme**: Tous les tests suivent le même pattern d'imports
2. **Setup cohérent**: `_project_root` ajouté au path de manière identique
3. **Gestion des dépendances**: `pytest.importorskip()` pour dépendances optionnelles (torch, pytorch_tabnet, shap, lime)
4. **Tests unitaires purs**: Chaque test isole une fonctionnalité spécifique
5. **Couverture complète**: Tests pour init, fit, predict, predict_proba, edge cases, erreurs
6. **Validation des interfaces**: Vérification de compatibilité sklearn

### ⚠️ Points d'Amélioration

1. **Fixtures partagées**: Certaines données (X, y) sont recréées dans chaque test - pourraient être des fixtures pytest
2. **Assertions détaillées**: Certaines assertions pourraient être plus explicites (messages d'erreur)
3. **Tests d'intégration**: Manquent tests end-to-end combinant plusieurs composants

---

## Recommandations

1. **Créer fixtures pytest** pour données communes:
   ```python
   @pytest.fixture
   def synthetic_binary_data():
       np.random.seed(42)
       X = np.random.randn(200, 10).astype(np.float32)
       y = np.random.randint(0, 2, 200)
       return X, y
   ```

2. **Ajouter tests de performance** (optionnel): Temps d'exécution, utilisation mémoire

3. **Tests de régression**: Comparer sorties avec versions précédentes (snapshots)

4. **Documentation inline**: Ajouter docstrings plus détaillées dans les tests

---

**Document créé automatiquement - À mettre à jour lors d'ajouts de tests**
