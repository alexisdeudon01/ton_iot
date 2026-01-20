import logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class PipelineValidator:
    """Phase 3: Validation (Hyperparameter tuning)"""

    def __init__(self, models, random_state=42):
        self.models = models
        self.random_state = random_state
        self.validation_results = {}

    def validate_tuning(self, X_val, y_val, output_dir):
        logger.info("[PHASE 3] Début de la validation (Tuning hyperparamètres)")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        X_val_num = X_val.select_dtypes(include=[np.number]).fillna(0)

        # Tuning pour RF (n_estimators)
        self._tune_algo('RF', X_val_num, y_val, 'n_estimators', [10, 50, 100, 200], output_path)

        # Tuning pour XGBoost (max_depth)
        self._tune_algo('XGBoost', X_val_num, y_val, 'max_depth', [3, 5, 7, 10], output_path)

        # Tuning pour DT (max_depth)
        self._tune_algo('DT', X_val_num, y_val, 'max_depth', [5, 10, 20, 30], output_path)

    def _tune_algo(self, name, X, y, param_name, values, output_path):
        logger.info(f"Tuning {name} sur {param_name}...")
        accs, f1s, aucs = [], [], []

        model = self.models.get(name)
        if model is None or name == 'CNN':
            return

        for val in values:
            try:
                model.set_params(**{param_name: val})
                model.fit(X, y)
                preds = model.predict(X)
                probs = model.predict_proba(X)[:, 1]

                accs.append(accuracy_score(y, preds))
                f1s.append(f1_score(y, preds))
                aucs.append(roc_auc_score(y, probs))
            except Exception as e:
                logger.warning(f"Tuning failed for {name} with {param_name}={val}: {e}")
                accs.append(0); f1s.append(0); aucs.append(0)

        plt.figure(figsize=(10, 6))
        plt.plot(values, accs, label='Accuracy', marker='o')
        plt.plot(values, f1s, label='F1-Score', marker='s')
        plt.plot(values, aucs, label='AUC', marker='^')
        plt.title(f"Validation Tuning: {name} ({param_name})")
        plt.xlabel(f"Variation de {param_name}")
        plt.ylabel("Scores")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / f"phase3_tuning_{name.lower()}.png")
        plt.close()

        self.validation_results[name] = {
            'best_param': values[np.argmax(f1s)],
            'best_f1': np.max(f1s)
        }
