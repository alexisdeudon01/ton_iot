import logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import pandas as pd
import numpy as np
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

class PipelineTester:
    """Phase 5: Testing (Final evaluation)"""

    def __init__(self, models):
        self.models = models
        self.test_results = {}

    def evaluate_all(self, X_test, y_test, output_dir):
        logger.info("[PHASE 5] Évaluation finale sur le Test Set")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        X_test_num = X_test.select_dtypes(include=[np.number]).fillna(0)

        metrics_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'AUC']

        for name, model in self.models.items():
            if model is None: continue
            logger.info(f"Évaluation finale de {name}...")

            try:
                if name == 'CNN':
                    model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X_test_num.values)
                        outputs = model(X_tensor)
                        _, predicted = torch.max(outputs.data, 1)
                        y_pred = predicted.numpy()
                        y_prob = torch.softmax(outputs, dim=1)[:, 1].numpy()
                else:
                    y_pred = model.predict(X_test_num)
                    y_prob = model.predict_proba(X_test_num)[:, 1]

                self.test_results[name] = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'F1-Score': f1_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred),
                    'Recall': recall_score(y_test, y_pred),
                    'AUC': roc_auc_score(y_test, y_prob)
                }
            except Exception as e:
                logger.error(f"Erreur lors de l'évaluation de {name}: {e}")

        self._plot_synthesis(output_path, metrics_names)
        self._generate_text_report(output_path)

    def _plot_synthesis(self, output_path, metrics_names):
        plt.figure(figsize=(12, 7))
        algos = list(self.test_results.keys())
        if not algos: return

        x = np.arange(len(algos))
        width = 0.15

        for i, metric in enumerate(metrics_names):
            scores = [self.test_results[algo][metric] for algo in algos]
            plt.bar(x + i*width, scores, width, label=metric)

        plt.title("Synthèse des Performances Finales par Algorithme")
        plt.xlabel("Algorithmes")
        plt.ylabel("Score")
        plt.xticks(x + width * 2, algos)
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.savefig(output_path / "phase5_final_synthesis.png")
        plt.close()

    def _generate_text_report(self, output_path):
        report_file = output_path / "final_report.txt"
        df_res = pd.DataFrame(self.test_results).T

        with open(report_file, 'w') as f:
            f.write("=== IRP DDOS DETECTION FINAL REPORT ===\n\n")
            f.write(df_res.to_string())
            if not df_res.empty:
                f.write("\n\nMeilleur modèle (F1-Score): " + str(df_res['F1-Score'].idxmax()))

        logger.info(f"Rapport final généré: {report_file}")
        print("\n" + "="*40)
        print("RÉSULTATS FINAUX")
        print("="*40)
        print(df_res)
