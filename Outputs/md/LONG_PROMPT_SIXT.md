# PROMPT COMPLET - TON IoT ML Pipeline DDoS Detection

## 🎯 CONTEXTE DU PROJET

Tu es un expert en Machine Learning et architecture logicielle travaillant sur un pipeline de détection DDoS pour l'IoT. Le projet analyse deux datasets majeurs (ToN-IoT et CIC-DDoS2019) avec plusieurs algorithmes ML (LR, DT, RF, KNN, CNN, TabNet).

### Architecture Actuelle

```
src/new_pipeline/
├── data_loader.py (188 lignes) - Chargement Dask lazy
├── trainer.py (187 lignes) - Entraînement 6 modèles
├── validator.py (98 lignes) - Tuning hyperparamètres
├── tester.py (127 lignes) - Évaluation finale
├── xai_manager.py (132 lignes) - Explainabilité (SHAP, LIME)
├── main.py (233 lignes) - Orchestration
└── config.py (52 lignes) - Configuration

Total: ~1017 lignes
```

### Technologies Utilisées

- **Dask**: Pour processing out-of-core (datasets 70M+ lignes)
- **PyTorch**: Pour CNN personnalisé sur données tabulaires
- **TabNet**: Réseau attentionnel pour données tabulaires
- **Scikit-learn**: Modèles traditionnels (LR, DT, RF, KNN)
- **SHAP/LIME**: Pour explainabilité modèles
- **SystemMonitor**: Thread background monitoring CPU/RAM

---

## 🔴 PROBLÈMES CRITIQUES IDENTIFIÉS

### CRITIQUE #1: Conversions Dask→Pandas Non Sécurisées

**Localisation**: `trainer.py:70-80`, `validator.py:27-30`, `tester.py:25-28`, `xai_manager.py:35-38`

**Problème**:

```python
# Code actuel - DANGEREUX
if isinstance(X_train, dd.DataFrame):
    X_train_pd = X_train.head(100000)  # ⚠️ Pas de vérification RAM!
    y_train_pd = y_train.head(100000)
```

**Impact**:

- Avec 50 colonnes × 100k lignes × 8 bytes = 40+ MB
- Peut exploser avec colonnes object/string
- **Cause des warnings RAM à 93%** visibles dans les logs
- Répété 3 fois (code dupliqué)

**Solution Requise**: Créer `MemoryAwareProcessor` qui:

1. Estime la taille mémoire du DataFrame
2. Vérifie la RAM disponible
3. Calcule un sample size sûr (≤70% RAM disponible)
4. Log toutes les décisions
5. Remplace les 4 occurrences de head()

---

### CRITIQUE #2: Gestion d'Erreurs Basique

**Localisation**: Partout dans le code

**Problème**:

```python
try:
    # ... code ...
except Exception as e:  # ⚠️ Trop large, masque les bugs
    logger.error(f"Erreur: {e}")  # Pas de retry, pas de contexte
```

**Occurrences**:

- `trainer.py:109` - Entraînement échoue silencieusement
- `validator.py:69` - Tuning échoue silencieusement
- `tester.py:70` - Évaluation échoue silencieusement
- `xai_manager.py:130` - XAI échoue silencieusement

**Impact**:

- Erreurs temporaires non récupérées
- Bugs masqués
- Debugging difficile

**Solution Requise**: Framework d'exceptions avec:

- Exceptions personnalisées typées
- Distinction erreurs recouvrables/critiques
- Retry automatique avec backoff exponentiel
- Contexte détaillé (model, phase, données)

---

### MOYEN #3: Retours de Fonctions Absents

**Problème**:

```python
def train_single(self, name, X_train, y_train):
    # ... train model ...
    # ❌ Pas de return - résultats dans self.models

def evaluate_all(self, X_test, y_test, algo_name=None):
    # ... evaluate ...
    # ❌ Résultats dans self.test_results
```

**Impact**:

- Impossible à tester unitairement
- État muté au lieu de retours explicites
- Pas de type hints
- Chaînage impossible

**Solution Requise**: Result Objects (dataclasses) avec:

- `TrainingResult`: model_name, success, time, history, error
- `ValidationResult`: model_name, best_params, scores
- `TestResult`: model_name, metrics (accuracy, f1, precision, recall, auc)
- Type hints explicites
- Méthodes helper (to_dict, from_dict)

---

### MINEUR #4: Visualisation Mélangée avec Logique

**Localisation**: `trainer.py:147-180` (40+ lignes matplotlib)

**Problème**: Classe Trainer fait trop de choses (violation Single Responsibility)

**Solution Requise**: `VisualizationService` centralisé avec méthodes:

- `plot_training_times(times: Dict[str, float])`
- `plot_convergence(name: str, history: Dict)`
- `plot_metrics_comparison(results: Dict)`
- `plot_confusion_matrices(cms: Dict)`
- `plot_resource_usage(monitor: SystemMonitor)`

---

### MINEUR #5: Configuration Sans Validation

**Localisation**: `config.py` (dicts simples)

**Problème**: Pas de validation, typos possibles, pas de types

**Solution Requise**: Pydantic Config avec:

- Validation automatique des paths
- Validation des plages (memory 10-90%)
- Type hints stricts
- Méthode `from_yaml()`
- Auto-complétion IDE

---

## 🎯 PLAN D'ACTION PRIORITAIRE (2 SEMAINES)

### SEMAINE 1: FIXES CRITIQUES

**Jour 1 (AUJOURD'HUI) - MemoryAwareProcessor**

```python
# Créer src/core/memory_manager.py
class MemoryAwareProcessor:
    def __init__(self, safety_margin: float = 0.7):
        self.safety_margin = safety_margin

    def safe_compute(self, dask_df: dd.DataFrame,
                     operation: str) -> pd.DataFrame:
        # 1. Estimer taille: n_rows * n_cols * 8 * 1.2
        # 2. Vérifier RAM disponible
        # 3. Si trop grand: sample avec ratio sûr
        # 4. Logger décision
        # 5. Retourner DataFrame pandas
        pass
```

**Intégration**:

- `trainer.py`: Remplacer lignes 70-74
- `validator.py`: Remplacer lignes 27-30
- `tester.py`: Remplacer lignes 25-28
- `xai_manager.py`: Remplacer lignes 35-38

**Tests**:

- `python3 main.py --test-mode` (0.1% data)
- Vérifier logs: "[MemoryAware] RAM suffisante, compute() complet"
- `python3 main.py --sample-ratio 0.5` (50% data)
- Vérifier RAM reste < 70%

---

**Jour 2 - Framework Exceptions**

```python
# Créer src/core/exceptions.py
class PipelineException(Exception):
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}

class ModelTrainingError(PipelineException): pass
class InsufficientMemoryError(PipelineException): pass
class DataLoadingError(PipelineException): pass
class ConfigurationError(PipelineException): pass
```

**Intégration**:

- Remplacer tous les `except Exception as e`
- Ajouter contexte dans details
- Propager erreurs critiques (raise)
- Logger erreurs avec extra=details

---

**Jour 3 - Result Objects**

```python
# Créer src/core/results.py
@dataclass
class TrainingResult:
    model_name: str
    success: bool
    training_time: float
    history: Dict[str, List[float]]
    error_message: Optional[str] = None

    @property
    def final_loss(self) -> float: ...
    @property
    def final_accuracy(self) -> float: ...

# Idem pour ValidationResult, TestResult, XAIResult
```

**Intégration**:

- `trainer.py`: `def train_single(...) -> TrainingResult`
- `validator.py`: `def validate_tuning(...) -> ValidationResult`
- `tester.py`: `def evaluate_all(...) -> Dict[str, TestResult]`
- Ajouter type hints partout

---

**Jour 4 - Tests & Validation**

- Tester pipeline complet end-to-end
- Vérifier RAM < 70% en continu
- Vérifier tous les Result objects
- Vérifier toutes les exceptions typées
- Benchmarker: temps, RAM peak, précision

---

**Jour 5 - Documentation**

- Mettre à jour README.md
- Documenter API (docstrings)
- Créer guide migration
- Ajouter exemples d'utilisation

---

### SEMAINE 2: REFACTORING ARCHITECTURE

**Jours 6-7 - VisualizationService**

```python
# Créer src/evaluation/visualization_service.py
class VisualizationService:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        plt.style.use('seaborn-v0_8')

    def plot_training_times(self, times: Dict[str, float]) -> Path: ...
    def plot_convergence(self, name: str, history: Dict) -> Path: ...
    def plot_metrics_comparison(self, results: Dict) -> Path: ...
```

**Intégration**:

- Extraire code matplotlib de `trainer.py` (lignes 147-180)
- Extraire de `validator.py`, `tester.py`, `xai_manager.py`
- Injecter `VisualizationService` dans constructeurs
- Supprimer code dupliqué

---

**Jours 8-9 - Pydantic Configuration**

```python
# Modifier src/new_pipeline/config.py
from pydantic import BaseModel, Field, validator

class PipelineConfig(BaseModel):
    # Paths
    ton_iot_path: Path = Field(...)
    cic_ddos_dir: Path = Field(...)

    # Algorithms
    algorithms: List[Literal['LR', 'DT', 'RF', 'KNN', 'CNN', 'TabNet']]

    # Resources
    max_memory_percent: float = Field(50.0, ge=10.0, le=90.0)

    @validator('ton_iot_path')
    def validate_path_exists(cls, v):
        if not v.exists():
            raise ValueError(f"Path not found: {v}")
        return v

    class Config:
        validate_assignment = True
```

**Intégration**:

- Remplacer dicts par PipelineConfig
- Mettre à jour `main.py`
- Créer `config.yaml` exemple
- Ajouter `config = PipelineConfig.from_yaml("config.yaml")`

---

**Jour 10 - Tests Finaux & Release**

- Tests d'intégration complets
- Tests avec full dataset (--sample-ratio 1.0)
- Monitoring RAM/CPU pendant 1h
- Créer release notes
- Git tag v2.0.0

---

## 📋 CODE TEMPLATES PRÊTS À UTILISER

### 1. MemoryAwareProcessor (PRIORITÉ #1)

```python
# src/core/memory_manager.py
import psutil
import logging
import dask.dataframe as dd
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

class MemoryAwareProcessor:
    """Convertit intelligemment Dask→Pandas selon RAM disponible"""

    def __init__(self, safety_margin: float = 0.7):
        """
        Args:
            safety_margin: Pourcentage RAM disponible à utiliser (0.7 = 70%)
        """
        self.safety_margin = safety_margin
        logger.info(f"[MemoryAware] Initialisé avec safety_margin={safety_margin*100:.0f}%")

    def safe_compute(self, dask_df: dd.DataFrame,
                     operation: str = "unknown") -> pd.DataFrame:
        """
        Convertit Dask→Pandas en respectant les limites RAM

        Args:
            dask_df: DataFrame Dask à convertir
            operation: Nom de l'opération (pour logging)

        Returns:
            DataFrame pandas (complet ou samplé)
        """
        # 1. Estimer taille en mémoire
        n_rows = len(dask_df)
        n_cols = len(dask_df.columns)

        # Estimation: 8 bytes/val numérique + 20% overhead
        estimated_bytes = n_rows * n_cols * 8 * 1.2
        estimated_mb = estimated_bytes / (1024 * 1024)

        # 2. Vérifier RAM disponible
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024 * 1024)
        safe_mb = available_mb * self.safety_margin

        logger.info(
            f"[MemoryAware] {operation}: "
            f"Estimé={estimated_mb:.1f}MB, "
            f"Disponible={available_mb:.1f}MB, "
            f"Safe={safe_mb:.1f}MB, "
            f"RAM actuelle={mem.percent:.1f}%"
        )

        # 3. Décision: compute ou sample
        if estimated_mb <= safe_mb:
            logger.info(f"[MemoryAware] ✓ RAM suffisante, compute() complet")
            return dask_df.compute()
        else:
            # Calculer ratio sûr
            safe_ratio = safe_mb / estimated_mb
            safe_rows = int(n_rows * safe_ratio)

            logger.warning(
                f"[MemoryAware] ⚠ RAM insuffisante! "
                f"Sampling {safe_rows:,} rows ({safe_ratio*100:.1f}%) "
                f"au lieu de {n_rows:,}"
            )

            # Échantillonnage stratifié si colonne target présente
            if "is_ddos" in dask_df.columns:
                return dask_df.sample(frac=safe_ratio, random_state=42).compute()
            else:
                return dask_df.head(safe_rows)

    def get_memory_status(self) -> dict:
        """Retourne état RAM actuel"""
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'percent_used': mem.percent,
            'safe_available_gb': (mem.available * self.safety_margin) / (1024**3)
        }

    def estimate_dataframe_size(self, n_rows: int, n_cols: int) -> float:
        """Estime taille DataFrame en MB"""
        return (n_rows * n_cols * 8 * 1.2) / (1024 * 1024)
```

**Intégration dans trainer.py**:

```python
# En haut du fichier
from src.core.memory_manager import MemoryAwareProcessor

class PipelineTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.memory_mgr = MemoryAwareProcessor(safety_margin=0.7)  # ✅ AJOUT
        self.models = {...}
        self.history = {}
        self.training_times = {}

    def train_single(self, name, X_train, y_train):
        """Trains a single model by name, handling Dask dataframes."""

        # REMPLACER ces lignes:
        # if isinstance(X_train, dd.DataFrame):
        #     X_train_pd = X_train.head(100000)
        #     y_train_pd = y_train.head(100000)

        # PAR:
        if isinstance(X_train, dd.DataFrame):
            X_train_pd = self.memory_mgr.safe_compute(X_train, f"training_{name}_X")
            y_train_pd = self.memory_mgr.safe_compute(y_train, f"training_{name}_y")
        else:
            X_train_pd = X_train
            y_train_pd = y_train

        # ... reste du code identique ...
```

---

### 2. Framework Exceptions

```python
# src/core/exceptions.py
"""Exceptions personnalisées pour le pipeline ML"""

class PipelineException(Exception):
    """Exception de base pour tout le pipeline"""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} [{details_str}]"
        return self.message

class DataLoadingError(PipelineException):
    """Erreur lors du chargement des données"""
    pass

class InsufficientMemoryError(PipelineException):
    """RAM insuffisante pour l'opération"""
    def __init__(self, required_mb: float, available_mb: float):
        super().__init__(
            f"RAM insuffisante: besoin {required_mb:.1f}MB, disponible {available_mb:.1f}MB",
            details={
                'required_mb': required_mb,
                'available_mb': available_mb,
                'deficit_mb': required_mb - available_mb
            }
        )

class ModelTrainingError(PipelineException):
    """Erreur lors de l'entraînement d'un modèle"""
    def __init__(self, model_name: str, original_error: Exception):
        super().__init__(
            f"Échec entraînement {model_name}: {str(original_error)}",
            details={
                'model': model_name,
                'error_type': type(original_error).__name__,
                'original_message': str(original_error)
            }
        )

class ValidationError(PipelineException):
    """Erreur lors de la validation"""
    pass

class ConfigurationError(PipelineException):
    """Erreur de configuration"""
    pass

class XAIError(PipelineException):
    """Erreur lors de l'analyse XAI"""
    pass
```

**Intégration dans trainer.py**:

```python
from src.core.exceptions import ModelTrainingError, InsufficientMemoryError

def train_single(self, name, X_train, y_train):
    start_time = time.time()

    try:
        # ... code entraînement ...

        self.training_times[name] = time.time() - start_time
        logger.info(f"{name} entraîné en {self.training_times[name]:.2f}s")

    except MemoryError as e:
        error = InsufficientMemoryError(
            required_mb=100,  # Estimation
            available_mb=psutil.virtual_memory().available / (1024**2)
        )
        logger.error(str(error), extra=error.details)
        raise error

    except Exception as e:
        error = ModelTrainingError(name, e)
        logger.error(str(error), extra=error.details)
        self.training_times[name] = 0
        raise error
```

---

### 3. Result Objects

```python
# src/core/results.py
"""Objets de résultats structurés pour le pipeline"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List, Any
import numpy as np
import json

@dataclass
class TrainingResult:
    """Résultat d'entraînement d'un modèle"""
    model_name: str
    success: bool
    training_time: float
    history: Dict[str, List[float]] = field(default_factory=dict)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def final_loss(self) -> float:
        """Dernière valeur de loss"""
        return self.history.get('loss', [float('inf')])[-1]

    @property
    def final_accuracy(self) -> float:
        """Dernière accuracy"""
        return self.history.get('accuracy', [0.0])[-1]

    def to_dict(self) -> dict:
        """Convertit en dict pour sérialisation"""
        return asdict(self)

    def to_json(self) -> str:
        """Convertit en JSON"""
        return json.dumps(self.to_dict(), indent=2)

@dataclass
class ValidationResult:
    """Résultat de validation hyperparamètres"""
    model_name: str
    best_params: Dict[str, Any]
    best_score: float
    all_scores: Dict[str, float] = field(default_factory=dict)
    validation_time: float = 0.0

    @property
    def improvement_percent(self) -> float:
        """Amélioration en % vs pire config"""
        if not self.all_scores:
            return 0.0
        worst = min(self.all_scores.values())
        return ((self.best_score - worst) / worst * 100) if worst > 0 else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class TestResult:
    """Résultat d'évaluation finale"""
    model_name: str
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    auc: float
    confusion_matrix: Optional[np.ndarray] = None
    test_time: float = 0.0

    def to_dict(self) -> dict:
        """Convertit en dict (sans confusion_matrix pour JSON)"""
        result = {
            'model': self.model_name,
            'accuracy': float(self.accuracy),
            'f1_score': float(self.f1_score),
            'precision': float(self.precision),
            'recall': float(self.recall),
            'auc': float(self.auc),
            'test_time': float(self.test_time)
        }
        if self.confusion_matrix is not None:
            result['confusion_matrix'] = self.confusion_matrix.tolist()
        return result

    @property
    def overall_score(self) -> float:
        """Score global (moyenne des métriques)"""
        return (self.accuracy + self.f1_score + self.precision +
                self.recall + self.auc) / 5

@dataclass
class XAIResult:
    """Résultat d'analyse XAI"""
    model_name: str
    method: str
    fidelity: float
    stability: float
    complexity: float
    composite_score: float = 0.0

    def __post_init__(self):
        """Calcule composite_score si non fourni"""
        if self.composite_score == 0.0:
            self.composite_score = (
                self.fidelity * 0.4 +
                self.stability * 0.4 +
                self.complexity * 0.2
            )

    def to_dict(self) -> dict:
        return asdict(self)
```

**Intégration dans trainer.py**:

```python
from src.core.results import TrainingResult

def train_single(self, name, X_train, y_train) -> TrainingResult:  # ✅ Type hint
    start_time = time.time()

    try:
        # ... code entraînement ...

        return TrainingResult(
            model_name=name,
            success=True,
            training_time=time.time() - start_time,
            history=self.history.get(name, {}),
            metadata={'n_samples': len(X_train_num)}
        )

    except Exception as e:
        return TrainingResult(
            model_name=name,
            success=False,
            training_time=time.time() - start_time,
            error_message=str(e)
        )
```

---

## 🧪 TESTS À EXÉCUTER

### Test 1: Petit Dataset (0.1%)

```bash
python3 main.py --test-mode
# Vérifier:
# - Pas d'erreur
# - RAM < 50%
# - Logs "[MemoryAware] ✓ RAM suffisante"
```

### Test 2: Dataset Moyen (50%)

```bash
python3 main.py --sample-ratio 0.5
# Vérifier:
# - RAM < 70%
# - Possibles "[MemoryAware] ⚠ Sampling"
# - Temps < 30min
```

### Test 3: Dataset Complet (100%)

```bash
python3 main.py --sample-ratio 1.0
# Vérifier:
# - Pas OOM
# - RAM < 80%
# - Logs sampling pour phases mémoire-intensives
```

### Test 4: Erreurs & Recovery

```bash
# Supprimer temporairement un fichier dataset
# Vérifier exception DataLoadingError levée
# Vérifier logs contexte détaillé
```

---

## 📊 MÉTRIQUES DE SUCCÈS

### Après Semaine 1

- ✅ Zéro OOM sur 5 runs consécutifs
- ✅ RAM reste < 70% durant tout le pipeline
- ✅ Toutes les exceptions typées (aucun `except Exception`)
- ✅ Toutes les fonctions retournent des Result objects
- ✅ Logs explicites: "[MemoryAware] ..." visible partout

### Après Semaine 2

- ✅ Code matplotlib séparé (VisualizationService)
- ✅ Configuration Pydantic validée
- ✅ Couverture tests > 60%
- ✅ Documentation API complète
- ✅ Guide migration écrit

---

## 🚨 RÈGLES CRITIQUES

### ❌ À NE JAMAIS FAIRE

1. **NE PAS** réécrire tout d'un coup - itératif seulement
2. **NE PAS** supprimer code avant que nouveau fonctionne
3. **NE PAS** commit sans tester
4. **NE PAS** toucher data_loader.py cette semaine (il fonctionne)
5. **NE PAS** oublier les type hints

### ✅ À TOUJOURS FAIRE

1. **TOUJOURS** tester avec `--test-mode` d'abord
2. **TOUJOURS** créer branche Git avant changement majeur
3. **TOUJOURS** logger les décisions importantes
4. **TOUJOURS** monitorer RAM pendant tests
5. **TOUJOURS** documenter les changements

---

## 📝 CHECKLIST JOURNALIÈRE

### Avant de Commencer

- [ ] Git branch créée (`git checkout -b fix/memory-aware`)
- [ ] Environment activé (`.toniot/bin/activate`)
- [ ] Tests existants passent (`python main_test.py`)

### Pendant le Développement

- [ ] Code écrit avec type hints
- [ ] Docstrings ajoutées
- [ ] Logs informatifs ajoutés
- [ ] Exceptions spécifiques utilisées

### Avant de Commit

- [ ] `python3 main.py --test-mode` passe
- [ ] RAM < 70% vérifié
- [ ] Logs propres (pas d'erreurs)
- [ ] Code formaté (black/autopep8)
- [ ] Git commit avec message descriptif

---

## 💡 PRIORITÉ ABSOLUE: MemoryAwareProcessor

**À FAIRE MAINTENANT**:

1. Créer fichier `src/core/memory_manager.py`
2. Copier code MemoryAwareProcessor complet ci-dessus
3. Intégrer dans `trainer.py` (ligne 47 et 70-74)
4. Tester: `python3 main.py --test-mode`
5. Vérifier logs "[MemoryAware] ..." apparaissent
6. Si OK, intégrer dans validator.py, tester.py, xai_manager.py

**Temps estimé**: 2-3 heures
**Impact**: Résout problème RAM critique immédiatement

---

## 🎓 RÉSUMÉ EXÉCUTIF

**État actuel**: Code fonctionnel mais fragile sur RAM
**Problème #1**: Conversions Dask→Pandas non contrôlées → RAM 93%
**Solution #1**: MemoryAwareProcessor (2-3h de travail)
**ROI attendu**: -50% risque OOM, +90% fiabilité

**Commence par là** → MemoryAwareProcessor → Teste → Puis passe au reste

Bon courage! 🚀
