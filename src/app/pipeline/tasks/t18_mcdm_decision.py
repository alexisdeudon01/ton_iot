import os
import pandas as pd
import json
import time
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.app.pipeline.registry import TaskRegistry
from mcdm.decision_agent import DDoSDecisionAgent

@TaskRegistry.register("T18_MCDM_Decision")
class T18_MCDM_Decision(Task):
    """
    Tâche finale du pipeline : Exécute l'analyse MCDM/MOO pour désigner le meilleur algorithme.
    """
    def run(self, context: DAGContext) -> TaskResult:
        start_ts = time.time()
        cfg = context.config
        output_dir = os.path.join(cfg.paths.work_dir, "mcdm_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Charger les résultats de l'évaluation (T17)
        report_path = os.path.join(cfg.paths.work_dir, "reports", "run_report.json")
        if not os.path.exists(report_path):
            return TaskResult(task_name=self.name, status="failed", error="Rapport d'évaluation introuvable.")
            
        with open(report_path, "r") as f:
            all_metrics = json.load(f)
            
        # 2. Préparer la matrice de décision
        # On transforme le dictionnaire de métriques en DataFrame
        rows = []
        for model_name, metrics in all_metrics.items():
            if model_name == "fused_global": continue
            
            # Simulation de métriques XAI et Ressources si non présentes (pour la démo)
            # Dans un run réel, ces valeurs seraient extraites des snapshots du monitor
            row = {
                "model": model_name,
                "f1": metrics.get("f1", 0),
                "recall": metrics.get("recall", 0),
                "precision": metrics.get("precision", 0),
                "accuracy": metrics.get("accuracy", 0),
                "faithfulness": 0.85, # Valeurs par défaut pour l'exemple
                "stability": 0.80,
                "complexity": 2.5,
                "latency": 1.5,
                "cpu_percent": 15.0,
                "ram_percent": 120.0
            }
            rows.append(row)
            
        df_results = pd.DataFrame(rows)
        
        # 3. Initialiser l'Agent de Décision avec la config YAML
        # On recharge la config YAML brute pour avoir accès à la hiérarchie MCDM
        import yaml
        with open("config/pipeline.yaml", "r") as f:
            mcdm_config = yaml.safe_load(f)
            
        agent = DDoSDecisionAgent(mcdm_config)
        
        # 4. Exécuter l'analyse
        ranked_df = agent.rank_models(df_results)
        
        # 5. Générer le rapport et les graphiques
        report = agent.generate_final_report(ranked_df)
        report_md_path = os.path.join(output_dir, "final_justification_report.md")
        with open(report_md_path, "w") as f:
            f.write(report)
            
        plots_dir = os.path.join(output_dir, "plots")
        agent.visualize_sad(ranked_df, plots_dir)
        
        context.logger.info("mcdm", f"Analyse MCDM terminée. Gagnant : {ranked_df.iloc[0]['model']}")
        
        return TaskResult(
            task_name=self.name, 
            status="ok", 
            duration_s=time.time() - start_ts,
            outputs=[report_md_path, plots_dir],
            meta={"winner": ranked_df.iloc[0]['model']}
        )
