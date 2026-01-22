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
        
        output_dir = os.path.join(cfg.paths.work_dir, "reports")
        os.makedirs(output_dir, exist_ok=True)
        
        context.logger.info("validating", "Evaluating only on split='test'")
        
        all_metrics = {}

        def compute_metrics(df: pl.DataFrame, name: str):
            # Filter for test split
            test_df = df.filter(pl.col("split") == "test")
            n_eval_rows = test_df.height
            
            if n_eval_rows == 0:
                context.logger.warning("validating", f"No test split rows for {name}")
                return None

            if cfg.test_mode and n_eval_rows < 200:
                context.logger.warning("validating", f"metrics unstable in test_mode for {name} (n={n_eval_rows})")

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

        # 1. Evaluate Per-Algorithm
        for path, ds_name in [(pred_cic_path, "cic"), (pred_ton_path, "ton")]:
            if os.path.exists(path):
                df_ds = context.table_io.read_parquet(path).collect()
                for algo in cfg.training.algorithms:
                    df_algo = df_ds.filter(pl.col("model") == algo)
                    if df_algo.height > 0:
                        m = compute_metrics(df_algo, f"{ds_name}_{algo}")
                        if m: all_metrics[f"{ds_name}_{algo}"] = m

        # 2. Evaluate Fused
        df_fused = context.table_io.read_parquet(pred_fused_art.path).collect()
        m_fused = compute_metrics(df_fused, "fused_global")
        if m_fused: all_metrics["fused_global"] = m_fused
        
        report_path = os.path.join(output_dir, "run_report.json")
        with open(report_path, "w") as f:
            json.dump(all_metrics, f, indent=4)
            
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
