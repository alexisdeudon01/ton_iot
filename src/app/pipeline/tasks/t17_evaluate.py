import os
import time
import json
import polars as pl
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.core.contracts.events import Event
from src.app.pipeline.registry import TaskRegistry

@TaskRegistry.register("T17_Evaluate")
class T17_Evaluate(Task):
    def run(self, context: DAGContext) -> TaskResult:
        from src.infra.resources.monitor import ResourceMonitor
        monitor = ResourceMonitor(context.event_bus, context.run_id)
        monitor.snapshot(self.name)
        
        start_ts = time.time()
        cfg = context.config
        
        pred_cic_path = os.path.join(cfg.paths.work_dir, "data", "predictions_cic.parquet")
        pred_ton_path = os.path.join(cfg.paths.work_dir, "data", "predictions_ton.parquet")
        pred_fused_art = context.artifact_store.load_prediction("predictions_fused")
        pred_fused_by_algo_path = os.path.join(cfg.paths.work_dir, "data", "predictions_fused_by_algo.parquet")
        
        output_dir = "reports"
        os.makedirs(output_dir, exist_ok=True)
        
        context.logger.info("validating", "Evaluating only on split='test'")
        
        # 1. Préparation des métadonnées globales de la méthodologie
        report_metadata = {
            "methodology": {
                "fusion_strategy": "Late Fusion (Averaging probabilities)",
                "sample_description": f"Stratified sampling (ratio={cfg.sample_ratio})",
                "k_folds": {
                    "enabled": False,
                    "description": "Fixed stratified split used instead of cross-validation for performance"
                },
                "data_splitting": {
                    "training": "70%",
                    "validation": "15%",
                    "testing": "15%",
                    "method": "Stratified Shuffle Split"
                }
            },
            "outputs": {
                "directories": {
                    "data_plots": "./graph/feature_distributions/",
                    "decision_plots": "./graph/decision/",
                    "processed_data": "./work/data/"
                },
                "generated_files": []
            }
        }

        all_metrics = {"_metadata": report_metadata}

        def compute_metrics(df: pl.DataFrame, name: str):
            # Filter for test split
            test_df = df.filter(pl.col("split") == "test")
            n_eval_rows = test_df.height
            
            if n_eval_rows == 0:
                context.logger.warning("validating", f"No test split rows for {name}")
                return None

            y_true = test_df["y_true"].to_numpy()
            y_proba = test_df["proba"].to_numpy()
            y_pred = (y_proba >= cfg.fusion.threshold).astype(int)
            
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred).tolist()
            
            # Metrics
            try:
                auc = float(roc_auc_score(y_true, y_proba))
            except:
                auc = 0.0

            # Robust balance extraction
            def get_balance_dict(df, col_name):
                counts = df.group_by(col_name).count().to_dicts()
                return {str(r[col_name]): int(r["count"]) for r in counts}

            res = {
                "methodology_context": report_metadata["methodology"], # Inclusion du contexte par algo
                "n_eval_rows": n_eval_rows,
                "y_true_balance": get_balance_dict(test_df, "y_true"),
                "y_pred_balance": get_balance_dict(test_df.with_columns(pl.lit(y_pred).alias("p")), "p"),
                "confusion_matrix": cm,
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "roc_auc": auc
            }
            
            context.logger.info("validating", f"Metrics for {name}", f1=res["f1"], accuracy=res["accuracy"])
            return res

        # 2. Evaluate Per-Algorithm
        for path, ds_name in [(pred_cic_path, "cic"), (pred_ton_path, "ton")]:
            if os.path.exists(path):
                df_ds = context.table_io.read_parquet(path).collect()
                for algo in cfg.training.algorithms:
                    df_algo = df_ds.filter(pl.col("model") == algo)
                    if df_algo.height > 0:
                        m = compute_metrics(df_algo, f"{ds_name}_{algo}")
                        if m: all_metrics[f"{ds_name}_{algo}"] = m

        # 3. Evaluate Fused (global)
        df_fused = context.table_io.read_parquet(pred_fused_art.path).collect()
        m_fused = compute_metrics(df_fused, "fused_global")
        if m_fused: all_metrics["fused_global"] = m_fused

        # 3b. Evaluate Fused per algorithm (if available)
        fused_by_algo_metrics = {}
        if os.path.exists(pred_fused_by_algo_path):
            df_fused_by_algo = context.table_io.read_parquet(pred_fused_by_algo_path).collect()
            for algo in cfg.training.algorithms:
                df_algo = df_fused_by_algo.filter(pl.col("model") == algo)
                if df_algo.height > 0:
                    m = compute_metrics(df_algo, f"fused_{algo}")
                    if m:
                        fused_by_algo_metrics[algo] = m
        
        # 4. Liste des graphiques de distribution générés
        dist_dir = os.path.join("graph", "feature_distributions")
        if os.path.exists(dist_dir):
            dist_files = []
            for root, _, files in os.walk(dist_dir):
                for fname in files:
                    if fname.endswith(".png"):
                        abs_path = os.path.abspath(os.path.join(root, fname))
                        dist_files.append({"name": fname, "path": abs_path, "type": "distribution_plot"})
            report_metadata["outputs"]["generated_files"].extend(dist_files)

        report_path = os.path.join(output_dir, "run_report.json")
        with open(report_path, "w") as f:
            json.dump(all_metrics, f, indent=4)
        
        context.logger.info("writing", f"Evaluation report saved to {report_path}")

        # 5. Export per-algorithm JSON configs
        algo_map = {a.key: a for a in cfg.algorithms}
        type_map = {
            "LR": "linear",
            "DT": "tree",
            "RF": "ensemble_tree",
            "CNN": "deep",
            "TabNet": "deep_tabular",
        }
        model_cfg_dir = "algorithm_configurations"
        os.makedirs(model_cfg_dir, exist_ok=True)

        for algo_key in cfg.training.algorithms:
            algo_cfg = algo_map.get(algo_key)
            results_cic = all_metrics.get(f"cic_{algo_key}")
            results_ton = all_metrics.get(f"ton_{algo_key}")
            results_fused = fused_by_algo_metrics.get(algo_key) if fused_by_algo_metrics else all_metrics.get("fused_global")

            payload = {
                "algorithm": algo_cfg.name if algo_cfg else algo_key,
                "type": type_map.get(algo_key, "unknown"),
                "hyperparameters": (algo_cfg.params if algo_cfg else {}),
                "training_config": {
                    "cv_folds": None,
                    "stratified": True,
                    "train_ratio": 0.7,
                },
                "results": {
                    "cic": results_cic or {},
                    "ton": results_ton or {},
                    "fused": results_fused or {},
                },
                "explainability": {
                    "shap_available": False,
                    "lime_available": False,
                    "feature_importance": {},
                },
                "resources": {
                    "training_time_sec": None,
                    "inference_time_ms": None,
                    "memory_mb": None,
                },
            }

            out_path = os.path.join(model_cfg_dir, f"{algo_key}.json")
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=4)
            
        context.event_bus.publish(Event(
            type="RUN_FINISHED",
            payload={
                "status": "ok",
                "total_time_s": time.time() - start_ts,
                "metrics": all_metrics,
                "report_path": report_path
            },
            ts=time.time(),
            run_id=context.run_id
        ))
        
        monitor.snapshot(self.name)
        return TaskResult(task_name=self.name, status="ok", duration_s=time.time() - start_ts, outputs=[report_path], meta=all_metrics)
