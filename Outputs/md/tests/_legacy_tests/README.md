# Legacy Tests

Ce dossier contient des tests anciens qui ont été déplacés pour éviter les conflits avec les nouveaux tests.

**⚠️ IMPORTANT:** Ces tests sont **exclus de la collecte pytest** via `pytest.ini` (norecursedirs).

## Pourquoi ce dossier existe ?

Pour éviter l'erreur pytest "import file mismatch" causée par :
- Des fichiers de test avec le même nom dans différents dossiers
- Exemple : `test_no_data_leakage.py` dans `tests/` et `tests/_oldtest/`

## Structure

- **Nouveau fichier actif** : `tests/test_no_data_leakage.py`
- **Ancien fichier (legacy)** : `tests/_legacy_tests/test_no_data_leakage.py` (ignoré par pytest)

## Utilisation

Ces tests ne sont **pas exécutés** par défaut. Si vous devez les exécuter :
```bash
pytest tests/_legacy_tests/  # Exécuter explicitement
```

## Migration

Les tests actifs et à jour se trouvent dans `tests/`. 
Les tests de ce dossier sont conservés uniquement pour référence historique.
