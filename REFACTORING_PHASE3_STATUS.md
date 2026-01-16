# Status Refactorisation Phase 3 - Soutenance-Ready

## ✅ Fichiers Créés

### Requirements
- `requirements-core.txt` - Dépendances minimales (sans torch/tabnet/shap/lime)
- `requirements-nn.txt` - Dépendances optionnelles (torch, pytorch-tabnet, shap, lime)

### CNN Optionnel (partiellement)
- `src/models_cnn.py` - ⚠️ Nécessite correction indentation (structure if TORCH_AVAILABLE)
- `src/main_pipeline.py` - CNN skip si torch absent (comme TabNet)

### Structure Phase 3 (à créer)
- `src/evaluation/__init__.py`
- `src/evaluation/metrics.py` - Calculs métriques Dimension 1
- `src/evaluation/resources.py` - Mesures Dimension 2 (time/ram/latency)
- `src/evaluation/visualizations.py` - Génération tous graphiques PNG
- `src/evaluation/reporting.py` - Génération rapports MD + INDEX.md

## ⚠️ TODO Critique

1. **Corriger models_cnn.py**: Structure if/else TORCH_AVAILABLE nécessite indentation correcte
2. **Créer modules evaluation/**: metrics.py, resources.py, visualizations.py, reporting.py
3. **Générer 27 graphiques PNG** avec noms exacts:
   - DIM 1: 8 graphiques (f1_bar, metrics_grouped, boxplot, confusion_matrix_*, roc_*, pr_*, heatmap)
   - DIM 2: 6 graphiques (train_time, peak_ram, latency, tradeoffs, pareto, heatmap)
   - DIM 3: 6 graphiques (score_bar, stacked, tradeoff, heatmap, shap_top, lime_top)
   - TRANSVERSAL: 5 graphiques (radar, scatter_*, scores_table)
4. **Créer CSV**: metrics_by_fold.csv, metrics_aggregated.csv, scores_normalized.csv
5. **Tests pytest**: test_requirements_behavior.py, test_visualizations_exist.py, test_resource_metrics_non_negative.py
6. **Mode synthetic**: CLI flag --synthetic pour Phase 3

## 📋 Commandes Actuelles

```bash
# Install
pip install -r requirements-core.txt
pip install -r requirements-nn.txt  # Optionnel

# Lancer
python main.py --phase 3

# Tests (à créer)
python -m pytest tests/test_requirements_behavior.py -v
```

## 🔄 État

- ✅ Requirements split (core/nn)
- ⚠️ CNN optionnel (code partiel, nécessite correction)
- ❌ Phase 3 complète (structure créée, modules à implémenter)

