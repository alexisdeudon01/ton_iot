# Rules for Sixt - TON IoT ML Pipeline

## 🎯 RÈGLES SYSTÈME GLOBALES

Ces règles guident Sixt dans toutes les interactions avec ce projet.

---

## 1. RÈGLES D'ARCHITECTURE

### 1.1 Séparation des Responsabilités

- **TOUJOURS** séparer logique métier et visualisation
- **TOUJOURS** utiliser des interfaces/protocols pour abstractions
- **TOUJOURS** favoriser composition over inheritance
- **NE JAMAIS** mélanger preprocessing et training dans une fonction
- **NE JAMAIS** mettre du code matplotlib dans une classe métier

### 1.2 Patterns de Design

- **TOUJOURS** utiliser Strategy Pattern pour algorithmes interchangeables
- **TOUJOURS** utiliser Factory Pattern pour création d'objets complexes
- **TOUJOURS** utiliser Observer Pattern pour événements/monitoring
- **PRÉFÉRER** Dependency Injection aux instanciations directes
- **PRÉFÉRER** dataclasses aux dicts pour structures de données

### 1.3 Structure de Fichiers

```
src/
├── core/           # Logique métier fondamentale
├── models/         # Définitions modèles ML
├── evaluation/     # Métriques et visualisation
├── new_pipeline/   # Pipeline principal
└── utils/          # Utilitaires génériques
```

---

## 2. RÈGLES DE GESTION MÉMOIRE

### 2.1 Dask DataFrame (Critiques)

- **TOUJOURS** garder opérations Dask lazy jusqu'au dernier moment
- **NE JAMAIS** faire `.compute()` sans vérifier RAM disponible
- **NE JAMAIS** utiliser `.head(N)` sans estimer taille mémoire
- **TOUJOURS** utiliser `MemoryAwareProcessor.safe_compute()`
- **TOUJOURS** logger décisions de sampling/compute

### 2.2 Conversions Dask→Pandas

```python
# ❌ INTERDIT
X_pd = dask_df.head(100000)

# ✅ OBLIGATOIRE
X_pd = memory_mgr.safe_compute(dask_df, "operation_name")
```

### 2.3 Limites Mémoire

- **Environnement Dev**: RAM < 50%
- **Environnement Test**: RAM < 70%
- **Environnement Prod**: RAM < 80%
- **TOUJOURS** monitorer avec SystemMonitor
- **ALERTER** si RAM > 85%

### 2.4 Sampling Stratégies

- **PRÉFÉRER** sampling stratifié (garde distribution classes)
- **TOUJOURS** utiliser `random_state=42` pour reproductibilité
- **TOUJOURS** documenter taille sample dans logs
- **CALCULER** sample size basé sur RAM disponible, pas valeur fixe

---

## 3. RÈGLES DE GESTION D'ERREURS

### 3.1 Exceptions (Critiques)

- **NE JAMAIS** utiliser `except Exception:` seul
- **TOUJOURS** utiliser exceptions personnalisées typées
- **TOUJOURS** logger contexte complet dans `extra=details`
- **TOUJOURS** propager erreurs critiques (`raise`)
- **DISTINCTION** erreurs recouvrables vs critiques

### 3.2 Types d'Exceptions

```python
# ✅ Exceptions à utiliser
from src.core.exceptions import (
    ModelTrainingError,      # Erreur training modèle
    InsufficientMemoryError, # RAM insuffisante
    DataLoadingError,        # Erreur chargement données
    ValidationError,         # Erreur validation
    ConfigurationError       # Erreur config
)

# ❌ À éviter
except Exception:  # Trop large
except:            # Pire encore
```

### 3.3 Retry Logic

- **TOUJOURS** retry erreurs recouvrables (max 3 fois)
- **TOUJOURS** utiliser backoff exponentiel (2^n secondes)
- **NE JAMAIS** retry erreurs de validation ou config
- **LOGGER** chaque tentative de retry

### 3.4 Context Logging

```python
# ✅ Bon logging
logger.error(
    f"Échec entraînement {model_name}",
    extra={
        'model': model_name,
        'error_type': type(e).__name__,
        'n_samples': len(X),
        'n_features': len(X.columns)
    }
)
```

---

## 4. RÈGLES DE RETOURS DE FONCTIONS

### 4.1 Result Objects (Obligatoire)

- **TOUJOURS** retourner Result objects (dataclasses)
- **NE JAMAIS** retourner None implicitement
- **NE JAMAIS** muter état global au lieu de retourner valeur
- **TOUJOURS** inclure flag `success: bool`

### 4.2 Types de Retours

```python
# ✅ Obligatoire
def train_single(...) -> TrainingResult:
    return TrainingResult(
        model_name=name,
        success=True,
        training_time=elapsed,
        history={'loss': [...], 'accuracy': [...]}
    )

# ❌ Interdit
def train_single(...):  # Pas de type hint
    self.models[name] = model  # Mutation état
    # Pas de return
```

### 4.3 Type Hints

- **TOUJOURS** ajouter type hints à tous paramètres et retours
- **UTILISER** `Optional[T]` pour valeurs nullables
- **UTILISER** `Union[T1, T2]` pour types multiples
- **UTILISER** `Dict[str, float]` plutôt que `dict`
- **TOUJOURS** importer depuis `typing`

---

## 5. RÈGLES DE LOGGING

### 5.1 Niveaux de Log

- **DEBUG**: Détails internes (variables, états)
- **INFO**: Progression normale (phase démarré/terminé)
- **WARNING**: Situations anormales mais gérables (sampling forcé)
- **ERROR**: Erreurs nécessitant attention (échec training)
- **CRITICAL**: Erreurs fatales (corruption données)

### 5.2 Format Messages

```python
# ✅ Format structuré
logger.info(f"[Phase 2] Training {model_name} started")
logger.info(f"[MemoryAware] RAM: {mem.percent:.1f}%, Action: {action}")
logger.warning(f"[Sampling] Forced to {n_samples:,} rows (RAM constraint)")

# ❌ Messages vagues
logger.info("Starting")  # Trop vague
logger.info(f"Error: {e}")  # Pas assez contexte
```

### 5.3 Informations Critiques à Logger

- **Décisions mémoire**: compute vs sample
- **Début/Fin phases**: avec timestamps
- **Paramètres importants**: sample_ratio, n_samples, etc.
- **Erreurs avec contexte**: model, phase, données
- **Métriques**: RAM%, CPU%, temps exécution

---

## 6. RÈGLES DE TESTS

### 6.1 Tests Requis

- **TOUJOURS** tester avec `--test-mode` avant commit
- **TOUJOURS** tester avec données réelles avant release
- **TOUJOURS** vérifier RAM < 70% pendant tests
- **TOUJOURS** valider outputs générés (graphiques, rapports)

### 6.2 Modes de Test

```bash
# Test rapide (obligatoire avant commit)
python3 main.py --test-mode  # 0.1% données, ~5min

# Test moyen (avant PR)
python3 main.py --sample-ratio 0.5  # 50% données, ~30min

# Test complet (avant release)
python3 main.py  # 100% données, ~2-3h
```

### 6.3 Assertions Critiques

- **VÉRIFIER** RAM < 70% en continu
- **VÉRIFIER** Zéro OOM errors
- **VÉRIFIER** Tous graphiques générés
- **VÉRIFIER** Logs sans erreurs (sauf warnings attendus)
- **VÉRIFIER** Métriques cohérentes (F1 > 0.5)

---

## 7. RÈGLES DE CONFIGURATION

### 7.1 Configuration Pydantic (Obligatoire)

- **TOUJOURS** utiliser Pydantic pour config
- **TOUJOURS** valider paths existent
- **TOUJOURS** valider plages numériques
- **NE JAMAIS** utiliser dicts simples pour config
- **TOUJOURS** utiliser type hints stricts

### 7.2 Validation Config

```python
# ✅ Avec validation
class PipelineConfig(BaseModel):
    max_memory_percent: float = Field(50.0, ge=10.0, le=90.0)

    @validator('ton_iot_path')
    def validate_exists(cls, v):
        if not v.exists():
            raise ValueError(f"Path not found: {v}")
        return v

# ❌ Sans validation
HYPERPARAMS = {
    'LR': {'C': [0.1, 1.0, 10.0]}  # Pas de validation
}
```

### 7.3 Valeurs par Défaut

- **random_state**: Toujours 42 (reproductibilité)
- **max_memory_percent**: 50% (sécurisé)
- **dask_workers**: 2 (équilibré)
- **sample_ratio**: 1.0 en prod, 0.001 en test

---

## 8. RÈGLES DE VISUALISATION

### 8.1 Service Centralisé

- **TOUJOURS** utiliser `VisualizationService`
- **NE JAMAIS** mettre matplotlib dans classe métier
- **TOUJOURS** sauvegarder avec `dpi=300, bbox_inches='tight'`
- **TOUJOURS** fermer figures avec `plt.close(fig)`

### 8.2 Conventions Graphiques

- **Style**: 'seaborn-v0_8'
- **Figure size**: (10, 6) pour plots standards
- **Colors**: Palette cohérente (skyblue, seaborn)
- **Grid**: `alpha=0.3` pour grilles discrètes
- **Title**: Toujours descriptif et clair

### 8.3 Nommage Fichiers

```python
# ✅ Nommage cohérent
"phase2_training_times.png"
"phase2_convergence_cnn.png"
"phase3_tuning_rf.png"

# ❌ Nommage incohérent
"plot1.png"
"results.png"
```

---

## 9. RÈGLES DE CODE QUALITY

### 9.1 Formatage

- **Longueur ligne**: Max 88 caractères (Black standard)
- **Imports**: Groupés (stdlib, third-party, local)
- **Docstrings**: Google style, toutes fonctions publiques
- **Comments**: Expliquer "pourquoi", pas "quoi"

### 9.2 Nommage

- **Classes**: PascalCase (`MemoryAwareProcessor`)
- **Fonctions**: snake_case (`safe_compute`)
- **Constantes**: UPPER_SNAKE_CASE (`MAX_MEMORY_PERCENT`)
- **Privé**: prefix underscore (`_internal_method`)

### 9.3 Complexité

- **Max lignes/fonction**: 50 (si plus, refactorer)
- **Max paramètres**: 5 (sinon utiliser dataclass)
- **Max imbrication**: 3 niveaux (utiliser early returns)
- **Cyclomatic complexity**: < 10

### 9.4 Documentation

- **Docstring**: Toutes classes et fonctions publiques
- **Type hints**: Tous paramètres et retours
- **Inline comments**: Pour logique complexe uniquement
- **README**: Maintenir à jour avec changements

---

## 10. RÈGLES GIT & WORKFLOW

### 10.1 Branches

```bash
main           # Production stable
dev            # Développement intégration
fix/*          # Corrections bugs
feature/*      # Nouvelles fonctionnalités
refactor/*     # Refactoring sans changement fonctionnel
```

### 10.2 Commits

- **Format**: `type(scope): message`
- **Types**: feat, fix, refactor, docs, test, chore
- **Message**: Impératif ("Add feature" pas "Added feature")
- **Taille**: Atomique, une chose à la fois

### 10.3 Pull Requests

- **TOUJOURS** créer PR pour changements majeurs
- **TOUJOURS** lancer tests avant PR
- **TOUJOURS** documenter changements dans PR description
- **TOUJOURS** lier issues si applicable

### 10.4 Avant Commit Checklist

- [ ] Tests passent (`python3 main.py --test-mode`)
- [ ] RAM < 70% vérifié
- [ ] Logs propres (pas d'erreurs)
- [ ] Code formaté (black/autopep8)
- [ ] Type hints ajoutés
- [ ] Docstrings à jour

---

## 11. RÈGLES DE PRIORITÉS

### 11.1 Ordre d'Importance

1. **Correctness**: Code doit être correct avant tout
2. **Memory Safety**: RAM management critique
3. **Error Handling**: Gestion erreurs robuste
4. **Maintainability**: Code compréhensible et modulaire
5. **Performance**: Optimisation si nécessaire

### 11.2 Compromis Acceptables

- **ACCEPTABLE**: Sacrifier 10% vitesse pour 50% moins RAM
- **ACCEPTABLE**: Code plus verbeux si plus clair
- **ACCEPTABLE**: Duplication minime si découplage
- **NON ACCEPTABLE**: Sacrifier sécurité RAM pour vitesse
- **NON ACCEPTABLE**: Code obscur "optimisé"

---

## 12. RÈGLES SPÉCIFIQUES PROJET

### 12.1 Datasets

- **ToN-IoT**: Fichier unique CSV (~17GB)
- **CIC-DDoS2019**: Multiple CSV dans sous-dossiers
- **TOUJOURS** utiliser Dask pour chargement
- **NE JAMAIS** charger tout en RAM d'un coup

### 12.2 Modèles

- **Sklearn**: LR, DT, RF, KNN (toujours disponibles)
- **PyTorch**: CNN (custom pour tabulaire)
- **TabNet**: Optionnel (graceful degradation si absent)
- **TOUJOURS** wrapper modèles dans interface commune

### 12.3 Features

- **Colonnes système**: `is_ddos`, `label`, `type`, `dataset`
- **Features**: Tout le reste (network flow stats)
- **TOUJOURS** filtrer colonnes système avant training
- **TOUJOURS** sélectionner uniquement numériques
- **TOUJOURS** remplir NaN avec fillna(0)

### 12.4 Métriques

- **Primaires**: F1-Score (équilibre precision/recall)
- **Secondaires**: Accuracy, Precision, Recall, AUC
- **Alertes**: F1 > 0.98 (overfitting), F1 < 0.5 (underfitting)
- **TOUJOURS** calculer les 5 métriques

---

## 13. ANTI-PATTERNS À ÉVITER

### 13.1 Memory Anti-Patterns

```python
# ❌ INTERDIT: Compute sans vérification
df = dask_df.compute()

# ❌ INTERDIT: head() avec valeur fixe
df = dask_df.head(100000)

# ❌ INTERDIT: Boucle sur DataFrame
for i, row in df.iterrows():  # Très lent
```

### 13.2 Error Handling Anti-Patterns

```python
# ❌ INTERDIT: Catch-all silencieux
try:
    risky_operation()
except:
    pass  # Erreur ignorée!

# ❌ INTERDIT: Exception trop large
except Exception:
    logger.error("Error")  # Pas assez contexte
```

### 13.3 Code Organization Anti-Patterns

```python
# ❌ INTERDIT: God class (fait trop de choses)
class Pipeline:
    def load(self): ...
    def train(self): ...
    def validate(self): ...
    def plot(self): ...  # Responsabilités mélangées

# ❌ INTERDIT: État global muté
GLOBAL_MODELS = {}  # État partagé dangereux
```

---

## 14. RÈGLES DE PERFORMANCE

### 14.1 Optimisations Autorisées

- **Vectorisation** avec NumPy/Pandas (toujours)
- **Parallelisation** avec n_jobs=-1 (sklearn)
- **Caching** résultats coûteux (avec LRU cache)
- **Dask lazy ops** (éviter computes prématurés)

### 14.2 Optimisations Interdites

- **NE PAS** optimiser prématurément
- **NE PAS** sacrifier clarté pour microsecondes
- **NE PAS** utiliser tricks obscurs
- **PROFILER** avant d'optimiser (pas de guessing)

### 14.3 Benchmarking

- **TOUJOURS** mesurer avant/après optimisation
- **TOUJOURS** utiliser time.time() pour timing
- **TOUJOURS** logger temps exécution phases
- **COMPARER** avec baseline avant/après

---

## 15. RÈGLES DE DÉPLOIEMENT

### 15.1 Environnements

- **Dev**: Machine locale, test-mode
- **Staging**: Serveur test, sample-ratio 0.5
- **Production**: Serveur prod, sample-ratio 1.0

### 15.2 Pré-Déploiement

- [ ] Tous tests passent (y compris test complet)
- [ ] Documentation à jour
- [ ] Changelog créé
- [ ] Git tag version (v2.0.0)
- [ ] Backup données et modèles

### 15.3 Monitoring Production

- **TOUJOURS** monitorer RAM/CPU en continu
- **TOUJOURS** logger dans fichiers (pas seulement stdout)
- **CONFIGURER** alertes si RAM > 85%
- **ARCHIVER** logs et résultats (30 jours min)

---

## 🎯 RÉSUMÉ DES RÈGLES CRITIQUES (TOP 10)

1. **NE JAMAIS** compute() Dask sans vérifier RAM → Utiliser `MemoryAwareProcessor`
2. **NE JAMAIS** utiliser `except Exception:` → Exceptions typées seulement
3. **TOUJOURS** retourner Result objects avec type hints
4. **TOUJOURS** tester avec `--test-mode` avant commit
5. **TOUJOURS** logger décisions importantes (sampling, erreurs)
6. **TOUJOURS** utiliser VisualizationService pour plots
7. **TOUJOURS** valider config avec Pydantic
8. **TOUJOURS** garder RAM < 70% (dev/test), < 80% (prod)
9. **TOUJOURS** documenter fonctions publiques (docstrings)
10. **TOUJOURS** suivre workflow Phase 1→2→3→4→5→6

---

Ces règles sont **NON NÉGOCIABLES** et doivent être respectées dans tout code généré, modifié ou reviewé.
