# Rapport d'Analyse de Bugs

**Date:** 2024-01-18  
**Analyse:** Révision complète du code pour identifier les bugs potentiels

---

## 🐛 Bugs Identifiés

### 1. ⚠️ **BUG CRITIQUE**: `transform_data()` peut échouer si `apply_scaling=False`

**Fichier:** `src/core/preprocessing_pipeline.py:737`

**Problème:**
```python
def transform_data(self, X: pd.DataFrame) -> np.ndarray:
    # ...
    # Scale
    X_scaled = self.scale_features(cast(np.ndarray, X_selected), fit=False)
    return X_scaled
```

Si `apply_scaling=False` dans `prepare_data()`, le scaler n'est jamais fitté (`scale_features()` avec `fit=True` n'est jamais appelé), mais `is_fitted=True` est défini. Ensuite, `transform_data()` appelle `scale_features(fit=False)` qui va vérifier `if not self.is_fitted` et lever une exception, OU essayer de transformer avec un scaler non-fitté.

**Impact:** `transform_data()` échouera avec un scaler non-fitté si `apply_scaling=False`.

**Solution:** Vérifier si le scaler doit être utilisé avant d'appeler `scale_features()`.

---

### 2. ⚠️ **BUG POTENTIEL**: Double vérification `feature_selection_k_dynamic`

**Fichier:** `src/phases/phase3_evaluation.py:362-373`

**Problème:**
- Ligne 362-364: `_get_preprocessing_profile()` calcule `feature_selection_k` dynamiquement si `feature_selection_k_dynamic=True` et supprime cette clé du profile.
- Ligne 372-373: `_apply_preprocessing_per_fold()` vérifie à nouveau `feature_selection_k_dynamic` qui n'existe plus.

**Impact:** La vérification dans `_apply_preprocessing_per_fold()` ne fonctionnera jamais car la clé a déjà été supprimée.

**Solution:** Supprimer la vérification redondante dans `_apply_preprocessing_per_fold()`.

---

### 3. ⚠️ **BUG POTENTIEL**: `transform_test()` - scaler non-fitté si `apply_scaling=False`

**Fichier:** `src/core/preprocessing_pipeline.py:796-803`

**Problème:**
Si `apply_scaling=False`, `is_fitted=True` mais `self.scaler` n'est jamais fitté. La condition `if self.is_fitted and self.scaler is not None and hasattr(self.scaler, "transform")` sera vraie, mais `self.scaler.transform()` échouera car le scaler n'est pas fitté.

**Impact:** Warning sera loggé mais le comportement est correct (retourne unscaled). Cependant, le try/except masque une erreur qui pourrait être évitée.

**Solution:** Améliorer la vérification pour ne pas essayer de transformer avec un scaler non-fitté.

---

### 4. ⚠️ **BUG MINEUR**: Import manquant dans `main()`

**Fichier:** `src/core/preprocessing_pipeline.py:859`

**Problème:**
```python
def main():
    from dataset_loader import DatasetLoader  # Import relatif incorrect
```

Devrait être un import absolu : `from src.core.dataset_loader import DatasetLoader`

**Impact:** Fonctionne seulement si le script est exécuté depuis le bon répertoire.

**Solution:** Utiliser un import absolu.

---

### 5. ⚠️ **BUG POTENTIEL**: `selected_features` peut être `None` dans `transform_data()`

**Fichier:** `src/core/preprocessing_pipeline.py:325-327`

**Problème:**
Si `apply_feature_selection=False`, `self.selected_features` n'est pas défini dans `select_features()` (seulement dans `prepare_data()` ligne 586). Dans `transform_data()`, ligne 731, on utilise `self.selected_features` qui pourrait ne pas être défini.

**Impact:** `AttributeError` si `transform_data()` est appelé avec un pipeline où `apply_feature_selection=False` et où `selected_features` n'a pas été défini.

**Solution:** Initialiser `self.selected_features` dans `__init__()` ou vérifier avant utilisation.

---

### 6. ⚠️ **BUG POTENTIEL**: Accès à `X.columns[i]` sans vérifier la validité de l'index

**Fichier:** `src/core/preprocessing_pipeline.py:308`

**Problème:**
```python
selected_indices = self.feature_selector.get_support(indices=True)
self.selected_features = [X.columns[i] for i in selected_indices]
```

Si `selected_indices` contient un index invalide (>= len(X.columns)), cela provoquera une `IndexError`.

**Impact:** Possible si le feature selector retourne des indices invalides (peu probable avec sklearn).

**Solution:** Vérifier que les indices sont valides.

---

### 7. ⚠️ **BUG MINEUR**: Gestion d'erreur trop large dans `transform_test()`

**Fichier:** `src/core/preprocessing_pipeline.py:797-800`

**Problème:**
Le `try/except` capture toutes les exceptions, masquant potentiellement d'autres erreurs que "scaler not fitted".

**Impact:** Debugging plus difficile si une autre erreur survient.

**Solution:** Capturer seulement les exceptions spécifiques (NotFittedError).

---

## ✅ Bugs Corrigés

Aucun bug critique corrigé jusqu'à présent dans cette analyse.

---

## 📋 Recommandations

1. **Tests unitaires supplémentaires** pour les cas limites:
   - `transform_data()` avec `apply_scaling=False`
   - `transform_test()` avec différents profils
   - Pipeline avec `apply_feature_selection=False`

2. **Validation des inputs** plus stricte:
   - Vérifier que `selected_indices` sont valides
   - Vérifier que `selected_features` est défini avant utilisation

3. **Meilleure gestion d'erreurs**:
   - Utiliser des exceptions spécifiques
   - Éviter les try/except trop larges

---

## 🔍 Tests à Exécuter

1. Test avec `apply_scaling=False` et `transform_data()`
2. Test avec `apply_feature_selection=False` et `transform_data()`
3. Test Phase 3 avec différents profils
4. Test avec indices invalides dans feature selection
