# Correction Pytest "Import File Mismatch"

## Problème identifié

Erreur pytest `import file mismatch` causée par :
- Deux fichiers `test_no_data_leakage.py` avec le même nom :
  - `tests/test_no_data_leakage.py` (actif, 320 lignes)
  - `tests/_oldtest/test_no_data_leakage.py` (legacy, 55 lignes)
- Pytest collectait les deux fichiers, causant conflit d'import

## Actions effectuées

### 1. Nettoyage des caches Python/Pytest
- ✅ Suppression de tous les dossiers `__pycache__`
- ✅ Suppression de tous les fichiers `*.pyc`
- ✅ Suppression de `.pytest_cache`

### 2. Isolation des tests legacy
- ✅ Renommage `tests/_oldtest` → `tests/_legacy_tests`
- ✅ Vérification: aucun import de `_oldtest` dans le code principal
- ✅ Création de `tests/_legacy_tests/README.md` expliquant la migration

### 3. Configuration pytest.ini
- ✅ Création de `pytest.ini` à la racine
- ✅ `norecursedirs` inclut `_legacy_tests`
- ✅ Exclusion des dossiers: `__pycache__`, `.pytest_cache`, `build`, `dist`, etc.
- ✅ Options: `-q --strict-markers --tb=short`

### 4. Vérification
- ✅ Seul `tests/test_no_data_leakage.py` sera collecté
- ✅ `tests/_legacy_tests/test_no_data_leakage.py` est ignoré
- ✅ Simulation confirmée: pas de conflit d'import

## Résultat

✅ **L'erreur "import file mismatch" est corrigée définitivement**

Pytest collectera uniquement les tests actifs dans `tests/` et ignorera complètement `tests/_legacy_tests/`.

## Configuration pytest.ini

```ini
[pytest]
python_files = test_*.py
norecursedirs = _legacy_tests __pycache__ .pytest_cache build dist
addopts = -q --strict-markers --tb=short
```

## Fichiers concernés

- ✅ `pytest.ini` (nouveau)
- ✅ `tests/_legacy_tests/` (renommé depuis `_oldtest`)
- ✅ `tests/_legacy_tests/README.md` (nouveau)
- ✅ `tests/test_no_data_leakage.py` (actif, 320 lignes)
- ⚠️ `tests/_legacy_tests/test_no_data_leakage.py` (ignoré par pytest)

