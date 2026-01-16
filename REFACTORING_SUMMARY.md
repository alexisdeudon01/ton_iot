# Résumé de la Refactorisation IRP Pipeline

## ✅ Fichiers Créés

### Configuration Centralisée
- `src/config.py`: `PipelineConfig` + `generate_108_configs()` (exactement 108 configs)

### Architecture App
- `src/app/cli.py`: Parsing args sans UI obligatoire
- `src/app/pipeline_runner.py`: Orchestration des 5 phases

### Phases
- `src/phases/phase1_config_search.py`: Phase 1 - Recherche des 108 configs
- `src/phases/phase2_apply_best_config.py`: Phase 2 - Application meilleure config (stub)
- `src/phases/phase3_evaluation.py`: Phase 3 - Évaluation 3D (stub)
- `src/phases/phase4_ahp_preferences.py`: Phase 4 - AHP Preferences (stub)
- `src/phases/phase5_topsis_ranking.py`: Phase 5 - TOPSIS Ranking (stub)

### Tests
- `tests/conftest.py`: Fixtures pytest (TEST_CONFIG)
- `tests/test_phase1_config_search.py`: Tests Phase 1 (108 configs)
- `tests/test_smoke_pipeline.py`: Smoke tests pipeline

### Entry Point
- `main.py`: Adapté pour utiliser nouveau CLI + runner

## 📋 Commandes

### Lancer le Pipeline
```bash
# Pipeline complet (toutes phases)
python main.py

# Phase spécifique
python main.py --phase 1    # Phase 1: Config search
python main.py --phase 2    # Phase 2: Apply best config
python main.py --phase 3    # Phase 3: Evaluation
python main.py --phase 4    # Phase 4: AHP
python main.py --phase 5    # Phase 5: TOPSIS

# Options
python main.py --test-mode        # Mode test (0.001% données)
python main.py --interactive      # UI Tkinter optionnelle
python main.py --output-dir out   # Dossier de sortie personnalisé
```

### Tests Pytest
```bash
# Tous les tests
python -m pytest tests/ -v

# Test spécifique
python -m pytest tests/test_phase1_config_search.py -v

# Smoke tests uniquement
python -m pytest tests/test_smoke_pipeline.py -v
```

### Vérifier les 108 Configs
```bash
python -c "import sys; sys.path.insert(0, 'src'); from config import generate_108_configs; configs = generate_108_configs(); print(f'✅ {len(configs)} configs générées')"
```

## 🔄 Compatibilité

- ✅ `python main.py` fonctionne toujours
- ✅ Ancien code préservé dans `src/main_pipeline.py` (pour référence)
- ⚠️  Phases 2-5 sont des stubs (à implémenter avec ancien code)

## 📝 TODO Restant

1. Implémenter Phase 2 avec ancien `IRPPipeline.phase1_preprocessing`
2. Implémenter Phase 3 avec ancien `IRPPipeline.phase3_evaluation` + ajouter inference latency/peak RAM
3. Implémenter Phase 4 avec gestion préférences AHP
4. Implémenter Phase 5 avec ancien `AHPTopsisFramework`
5. Déplacer modules core dans `src/core/` (loader, harmonization, preprocessing)
6. Ajouter tests harmonization + metrics
7. Mettre à jour README.md avec architecture 5 phases

## 🏗️ Architecture Cible

```
src/
├── config.py              # Configuration centralisée
├── app/
│   ├── cli.py            # CLI parsing
│   └── pipeline_runner.py # Orchestration
├── phases/
│   ├── phase1_config_search.py
│   ├── phase2_apply_best_config.py
│   ├── phase3_evaluation.py
│   ├── phase4_ahp_preferences.py
│   └── phase5_topsis_ranking.py
└── core/                  # (À créer)
    ├── dataset_loader.py
    ├── data_harmonization.py
    └── preprocessing_pipeline.py

tests/
├── conftest.py
├── test_phase1_config_search.py
└── test_smoke_pipeline.py

main.py                    # Entry point
