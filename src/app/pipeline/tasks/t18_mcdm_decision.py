import os
import pandas as pd
import json
import time
import yaml
import joblib
import numpy as np
import polars as pl
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from src.mcdm.metrics_utils import compute_f_perf, compute_f_expl
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

        dtreeviz_dir = os.path.join("graph", "algorithms", "dtreeviz")
        dtreeviz_files = self._generate_dtreeviz(context, dtreeviz_dir)

        sampling_summary = self._generate_sampling_variation(all_metrics)
        report_links = self._write_graphs_report(sampling_summary)

        # 6. Mise à jour du rapport JSON avec les sorties MCDM et liens vers les graphiques complets
        if "_metadata" in all_metrics:
            # Ajout des fichiers de décision
            decision_files = []
            for root, _, files in os.walk(plots_dir):
                for f in files:
                    if not f.endswith(".png"):
                        continue
                    path = os.path.abspath(os.path.join(root, f))
                    decision_files.append({
                        "name": f,
                        "path": path,
                        "url": f"file://{path}",
                        "type": "decision_plot"
                    })
            algo_viz_files = [
                {
                    "name": os.path.basename(p),
                    "path": os.path.abspath(p),
                    "url": f"file://{os.path.abspath(p)}",
                    "type": "algorithm_viz"
                }
                for p in dtreeviz_files
            ]
            variation_reports = []
            variations_dir = os.path.join("reports", "variations")
            if os.path.exists(variations_dir):
                for root, _, files in os.walk(variations_dir):
                    for f in files:
                        if not (f.endswith(".docx") or f.endswith(".md")):
                            continue
                        path = os.path.abspath(os.path.join(root, f))
                        variation_reports.append({
                            "name": f,
                            "path": path,
                            "url": f"file://{path}",
                            "type": "variation_report"
                        })
            all_metrics["_metadata"]["outputs"]["generated_files"].extend(decision_files)
            all_metrics["_metadata"]["outputs"]["generated_files"].extend(algo_viz_files)
            all_metrics["_metadata"]["outputs"]["generated_files"].extend(variation_reports)
            for entry in report_links:
                all_metrics["_metadata"]["outputs"]["generated_files"].append(entry)
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

    def _generate_dtreeviz(self, context: DAGContext, output_dir: str) -> list:
        os.makedirs(output_dir, exist_ok=True)
        try:
            from dtreeviz.trees import dtreeviz
        except Exception as exc:
            context.logger.warning("mcdm", f"dtreeviz not available: {exc}")
            return []

        files = []
        dataset_specs = [
            ("cic", "cic_projected", "preprocess_cic.joblib", "cic_DT.model", "DT"),
            ("ton", "ton_projected", "preprocess_ton.joblib", "ton_DT.model", "DT"),
            ("cic", "cic_projected", "preprocess_cic.joblib", "cic_RF.model", "RF"),
            ("ton", "ton_projected", "preprocess_ton.joblib", "ton_RF.model", "RF"),
        ]

        for dataset, table_id, prep_name, model_name, algo in dataset_specs:
            model_path = os.path.join("work", "models", model_name)
            prep_path = os.path.join("work", "artifacts", prep_name)

            if not os.path.exists(model_path):
                context.logger.warning("mcdm", f"Model missing for dtreeviz: {model_path}")
                continue

            try:
                table_art = context.artifact_store.load_table(table_id)
            except Exception as exc:
                context.logger.warning("mcdm", f"Table missing for dtreeviz: {table_id} ({exc})")
                continue

            if not os.path.exists(table_art.path):
                context.logger.warning("mcdm", f"Table path missing: {table_art.path}")
                continue

            df = context.table_io.read_parquet(table_art.path).collect().to_pandas()
            if df.empty:
                context.logger.warning("mcdm", f"Empty table for dtreeviz: {table_id}")
                continue

            if "y" in df.columns:
                y = df["y"].fillna(0).astype(int)
            else:
                y = pd.Series(np.zeros(len(df), dtype=int))

            if len(df) > 500:
                df = df.sample(n=500, random_state=42)
                y = y.loc[df.index]

            model_bundle = joblib.load(model_path)
            model = model_bundle.get("model")
            feature_order = model_bundle.get("feature_order", [c for c in df.columns if c != "y"])

            X_raw = df[feature_order]
            feature_names = feature_order
            if os.path.exists(prep_path):
                try:
                    preprocessor = joblib.load(prep_path)
                    X = preprocessor.transform(X_raw)
                    try:
                        feature_names = list(preprocessor.get_feature_names_out(feature_order))
                    except Exception:
                        feature_names = feature_order
                except Exception as exc:
                    context.logger.warning("mcdm", f"Preprocess load failed for {prep_path}: {exc}")
                    X = X_raw.values
            else:
                X = X_raw.values

            tree_model = model
            if algo == "RF":
                try:
                    tree_model = model.estimators_[0]
                except Exception:
                    context.logger.warning("mcdm", f"No estimator available in RF for {model_path}")
                    continue

            class_names = [str(c) for c in sorted(y.unique())]
            try:
                viz = dtreeviz(
                    tree_model,
                    X,
                    y.values,
                    target_name="label",
                    feature_names=feature_names,
                    class_names=class_names
                )
                out_name = f"{dataset.lower()}_{algo.lower()}_dtreeviz.svg"
                out_path = os.path.join(output_dir, out_name)
                viz.save(out_path)
                files.append(out_path)
            except Exception as exc:
                context.logger.warning("mcdm", f"dtreeviz failed for {model_path}: {exc}")
                continue

        return files
