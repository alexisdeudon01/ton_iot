import os
import pandas as pd
import json
import time
import yaml
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.app.pipeline.registry import TaskRegistry
from mcdm.decision_agent import DDoSDecisionAgent

@TaskRegistry.register("T18_MCDM_Decision")
class T18_MCDM_Decision(Task):
    """
    Tâche finale du pipeline : Exécute l'analyse MCDM/MOO pour désigner le meilleur algorithme.
    Met à jour le rapport JSON avec les sorties décisionnelles et les liens vers les graphiques.
    """
    def run(self, context: DAGContext) -> TaskResult:
        start_ts = time.time()
        cfg = context.config
        output_dir = "reports"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Charger les résultats de l'évaluation (T17)
        report_path = os.path.join("reports", "run_report.json")
        if not os.path.exists(report_path):
            return TaskResult(task_name=self.name, status="failed", duration_s=time.time()-start_ts, error="Rapport d'évaluation introuvable.")
            
        with open(report_path, "r") as f:
            all_metrics = json.load(f)
            
        # 2. Préparer la matrice de décision
        rows = []
        for model_name, metrics in all_metrics.items():
            if model_name == "_metadata": continue
            if not model_name.startswith("fused_"):
                continue
            if model_name == "fused_global":
                continue
            
            mcdm_inputs = metrics.get("mcdm_inputs", {})
            shap_std = mcdm_inputs.get("shap_std", 0.5)
            row = {
                "model": model_name,
                "f1": metrics.get("f1", 0),
                "recall": metrics.get("recall", 0),
                "precision": metrics.get("precision", 0),
                "accuracy": metrics.get("accuracy", 0),
                "faithfulness": metrics.get("faithfulness", mcdm_inputs.get("s_intrinsic", 0.0)),
                "stability": metrics.get("stability", max(0.0, 1.0 - shap_std)),
                "complexity": metrics.get("complexity", mcdm_inputs.get("n_params", 1000)),
                "latency": metrics.get("latency", 1.0),
                "cpu_percent": metrics.get("cpu_percent", 0.0),
                "ram_percent": metrics.get("ram_percent", 0.0)
            }
            rows.append(row)
            
        df_results = pd.DataFrame(rows)
        
        # 3. Initialiser l'Agent de Décision
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
            
        plots_dir = os.path.join("graph", "decision")
        os.makedirs(plots_dir, exist_ok=True)
        agent.visualize_sad(ranked_df, plots_dir)

        # 6. Mise à jour du rapport JSON avec les sorties MCDM et liens vers les graphiques complets
        if "_metadata" in all_metrics:
            # Ajout des fichiers de décision
            decision_files = [
                {
                    "name": f, 
                    "path": os.path.abspath(os.path.join(plots_dir, f)), 
                    "url": f"file://{os.path.abspath(os.path.join(plots_dir, f))}",
                    "type": "decision_plot"
                }
                for f in os.listdir(plots_dir) if f.endswith(".png")
            ]
            all_metrics["_metadata"]["outputs"]["generated_files"].extend(decision_files)
            all_metrics["_metadata"]["outputs"]["final_report_markdown"] = {
                "path": os.path.abspath(report_md_path),
                "url": f"file://{os.path.abspath(report_md_path)}"
            }
            
            with open(report_path, "w") as f:
                json.dump(all_metrics, f, indent=4)
        
        context.logger.info("mcdm", f"Analyse MCDM terminée. Rapport : {report_md_path}")
        
        return TaskResult(
            task_name=self.name, 
            status="ok", 
            duration_s=time.time() - start_ts,
            outputs=[report_md_path, plots_dir],
            meta={"winner": ranked_df.iloc[0]['model']}
        )
