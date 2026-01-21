import logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
from pathlib import Path
from src.new_pipeline.config import HYPERPARAMS

logger = logging.getLogger(__name__)

class PipelineValidator:
    """Phase 3: Validation (Tuning Dynamique Hors Config)"""

    def __init__(self, models, random_state=42):
        self.models = models
        self.random_state = random_state
        self.best_params = {}

    def validate_tuning(self, X_val, y_val, rr_dir: Path):
        logger.info("[PHASE 3] Début de la validation (Tuning hyperparamètres)")

        X_val_num = X_val.select_dtypes(include=[np.number]).fillna(0)

        for name in ['LR', 'DT', 'RF']:
            if name not in self.models: continue

            # On récupère la grille depuis la config
            grid = HYPERPARAMS.get(name, {})
            if not grid: continue

            # Pour la démo, on tune le premier paramètre de la grille
            param_name = list(grid.keys())[0]
            param_values = grid[param_name]

            self._tune_algo(name, X_val_num, y_val, param_name, param_values, rr_dir)

    def _tune_algo(self, name, X, y, param_name, values, rr_dir):
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

        # Graphique par algorithme (X: Variation, Y: Scores)
        plt.figure(figsize=(10, 6))
        plt.plot(values, accs, label='Accuracy', marker='o')
        plt.plot(values, f1s, label='F1-Score', marker='s')
        plt.plot(values, aucs, label='AUC', marker='^')
        plt.title(f"Validation Tuning: {name} ({param_name})")
        plt.xlabel(f"Variation de {param_name}")
        plt.ylabel("Scores")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(rr_dir / f"phase3_tuning_{name.lower()}.png")
        plt.close()

        # Sélection du meilleur
        best_idx = np.argmax(f1s)
        self.best_params[name] = {param_name: values[best_idx]}
        logger.info(f"[RESULT PHASE 3] Meilleur {param_name} pour {name}: {values[best_idx]}")
