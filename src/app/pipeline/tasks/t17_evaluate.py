import os
import time
import json
import polars as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
        
        # Load individual predictions to evaluate per-algo performance
        pred_cic_path = os.path.join(context.config.paths.work_dir, "data", "predictions_cic.parquet")
        pred_ton_path = os.path.join(context.config.paths.work_dir, "data", "predictions_ton.parquet")
        pred_fused_art = context.artifact_store.load_prediction("predictions_fused")
        
        output_dir = os.path.join(context.config.paths.work_dir, "reports")
        os.makedirs(output_dir, exist_ok=True)
        
        context.logger.info("validating", "Evaluating all algorithms independently + Fused result")
        
        all_metrics = {}

        def compute_metrics(df: pl.DataFrame, name: str):
            y_true = df["y_true"].to_numpy()
            y_proba = df["proba"].to_numpy()
            y_pred = (y_proba >= context.config.fusion.threshold).astype(int)
            
            # Handle cases with only one class in y_true for ROC AUC
            try:
                auc = float(roc_auc_score(y_true, y_proba))
            except:
                auc = 0.0

            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "roc_auc": auc
            }

        # 1. Evaluate Per-Algorithm
        for path, ds_name in [(pred_cic_path, "cic"), (pred_ton_path, "ton")]:
            if os.path.exists(path):
                df_ds = context.table_io.read_parquet(path).collect()
                for algo in context.config.training.algorithms:
                    df_algo = df_ds.filter(pl.col("model") == algo)
                    if df_algo.height > 0:
                        all_metrics[f"{ds_name}_{algo}"] = compute_metrics(df_algo, f"{ds_name}_{algo}")

        # 2. Evaluate Fused
        df_fused = context.table_io.read_parquet(pred_fused_art.path).collect()
        all_metrics["fused_global"] = compute_metrics(df_fused, "fused_global")
        
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
        
        context.logger.info("validating", f"Evaluation complete. Fused F1={all_metrics['fused_global']['f1']:.4f}", 
                            report_path=report_path)
        
        monitor.snapshot(self.name)
        return TaskResult(
            task_name=self.name,
            status="ok",
            duration_s=time.time() - start_ts,
            outputs=[report_path],
            meta=all_metrics
        )
