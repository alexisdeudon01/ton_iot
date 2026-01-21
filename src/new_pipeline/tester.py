import logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import torch
import dask.dataframe as dd

logger = logging.getLogger(__name__)

class PipelineTester:
    """Phase 5: Testing (Évaluation Finale) with Dask support"""

    def __init__(self, models, rr_dir: Path):
        self.models = models
        self.rr_dir = rr_dir
        self.test_results = {}

    def evaluate_all(self, X_test, y_test, algo_name: Optional[str] = None):
        logger.info("[PHASE 5] Évaluation finale sur le jeu de Test")

        # Handle Dask dataframes by sampling for evaluation
        if isinstance(X_test, dd.DataFrame):
            print(f"INFO: Conversion Dask -> Pandas pour l'évaluation (Sampling 100k rows)")
            X_test_pd = X_test.head(100000)
            y_test_pd = y_test.head(100000)
        else:
            X_test_pd = X_test
            y_test_pd = y_test

        X_test_num = X_test_pd.select_dtypes(include=[np.number]).fillna(0)
        metrics_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'AUC']

        algos_to_eval = [algo_name] if algo_name else list(self.models.keys())

        for name in algos_to_eval:
            model = self.models.get(name)
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
                elif name == 'TabNet':
                    y_pred = model.predict(X_test_num.values)
                    y_prob = model.predict_proba(X_test_num.values)[:, 1]
                else:
                    y_pred = model.predict(X_test_num)
                    y_prob = model.predict_proba(X_test_num)[:, 1]

                self.test_results[name] = {
                    'Accuracy': accuracy_score(y_test_pd, y_pred),
                    'F1-Score': f1_score(y_test_pd, y_pred),
                    'Precision': precision_score(y_test_pd, y_pred),
                    'Recall': recall_score(y_test_pd, y_pred),
                    'AUC': roc_auc_score(y_test_pd, y_prob)
                }

                # Overfitting/Underfitting Analysis
                self._analyze_fit(name, self.test_results[name]['F1-Score'])

            except Exception as e:
                logger.error(f"  Erreur évaluation {name}: {e}")

        self._plot_final_synthesis(metrics_names)
        self._generate_final_report()

    def _analyze_fit(self, name: str, test_f1: float):
        """Analyzes if the model is overfitting or underfitting."""
        print(f"\nANALYSE DE FIT POUR {name}:")
        # In a real scenario, we would compare with training F1
        # For this expert pipeline, we use a heuristic based on test performance
        if test_f1 > 0.98:
            print(f"  ⚠️  ALERTE: Overfitting potentiel suspecté (F1={test_f1:.4f})")
        elif test_f1 < 0.5:
            print(f"  ⚠️  ALERTE: Underfitting suspecté (F1={test_f1:.4f})")
        else:
            print(f"  ✅ Fit correct (F1={test_f1:.4f})")

    def _plot_final_synthesis(self, metrics_names):
        """Génère un graphique de synthèse (X: Algos, Y: Score) avec une courbe par métrique."""
        plt.figure(figsize=(12, 7))

        algos = list(self.test_results.keys())
        if not algos: return

        for metric in metrics_names:
            scores = [self.test_results[algo][metric] for algo in algos]
            plt.plot(algos, scores, marker='o', label=metric, linewidth=2)

        plt.title("Synthèse des Performances Finales (Comparaison des Algos)")
        plt.xlabel("Algorithmes")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(self.rr_dir / "phase5_final_synthesis.png")
        plt.close()

    def _generate_final_report(self):
        """Génère le rapport texte final."""
        report_file = self.rr_dir / "final_report_expert.txt"
        df_res = pd.DataFrame(self.test_results).T

        with open(report_file, 'w') as f:
            f.write("===========================================\n")
            f.write("IRP DDOS DETECTION - RAPPORT FINAL EXPERT\n")
            f.write("===========================================\n\n")
            f.write(df_res.to_string())
            if not df_res.empty:
                best_model = df_res['F1-Score'].idxmax()
                f.write(f"\n\nMODÈLE RECOMMANDÉ : {best_model} (F1-Score: {df_res.loc[best_model, 'F1-Score']:.4f})")

        logger.info(f"[SUCCESS] Rapport final généré: {report_file}")
        print("\n" + "="*50)
        print("RÉSULTATS FINAUX CONSOLIDÉS")
        print("="*50)
        print(df_res)
        print("="*50)
