import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import dask.dataframe as dd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.new_pipeline.config import HYPERPARAMS
from src.core.memory_manager import MemoryAwareProcessor
from src.core.exceptions import ValidationError
from src.core.results import ValidationResult
from src.evaluation.visualization_service import VisualizationService

logger = logging.getLogger(__name__)

class PipelineValidator:
    """Phase 3: Validation (Tuning Dynamique) with Memory Safety"""

    def __init__(self, models, memory_mgr: MemoryAwareProcessor, viz_service: VisualizationService, random_state=42):
        self.models = models
        self.memory_mgr = memory_mgr
        self.viz = viz_service
        self.random_state = random_state
        self.best_params = {}

    def validate_tuning(self, X_val, y_val, algo_name: Optional[str] = None) -> Dict[str, ValidationResult]:
        logger.info("[PHASE 3] Début de la validation (Tuning hyperparamètres)")
        results = {}

        # 1. Memory Safe Conversion
        if isinstance(X_val, dd.DataFrame):
            X_val_pd = self.memory_mgr.safe_compute(X_val, "validation_tuning")
            y_val_pd = self.memory_mgr.safe_compute(y_val, "validation_tuning_labels")
        else:
            X_val_pd, y_val_pd = X_val, y_val

        X_val_num = X_val_pd.select_dtypes(include=[np.number]).fillna(0)

        algos_to_tune = [algo_name] if algo_name else ['LR', 'DT', 'RF', 'KNN']

        for name in algos_to_tune:
            if name not in self.models or self.models[name] is None: continue

            grid = HYPERPARAMS.get(name, {})
            if not grid: continue

            param_name = list(grid.keys())[0]
            param_values = grid[param_name]

            results[name] = self._tune_algo(name, X_val_num, y_val_pd, param_name, param_values)
        
        return results

    def _tune_algo(self, name, X, y, param_name, values) -> ValidationResult:
        logger.info(f"Tuning dynamique de {name} sur {param_name}...")
        accs, f1s, aucs = [], [], []

        model = self.models[name]

        for val in values:
            try:
                # Mise à jour dynamique de l'hyperparamètre
                model.set_params(**{param_name: val})
                model.fit(X, y)

                preds = model.predict(X)
                probs = model.predict_proba(X)[:, 1]

                accs.append(accuracy_score(y, preds))
                f1s.append(f1_score(y, preds))
                aucs.append(roc_auc_score(y, probs))

                logger.info(f"  {param_name}={val} -> F1: {f1s[-1]:.4f}")
            except Exception as e:
                logger.warning(f"  Échec tuning {name} ({val}): {e}")
                accs.append(0); f1s.append(0); aucs.append(0)

        # Visualisation via service
        plot_path = self.viz.plot_tuning_results(name, param_name, values, accs, f1s, aucs)

        # Sélection du meilleur
        best_idx = np.argmax(f1s)
        best_val = values[best_idx]
        self.best_params[name] = {param_name: best_val}
        logger.info(f"[RESULT PHASE 3] Meilleur {param_name} pour {name}: {best_val}")
        
        return ValidationResult(
            model_name=name,
            best_params={param_name: best_val},
            best_score=f1s[best_idx],
            tuning_plot_path=plot_path
        )
