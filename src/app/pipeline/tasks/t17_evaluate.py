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
        pred_art = context.artifact_store.load_prediction("predictions_fused")
        
        output_dir = os.path.join(context.config.paths.work_dir, "reports")
        os.makedirs(output_dir, exist_ok=True)
        
        context.logger.info("validating", "Evaluating fused predictions")
        
        df = context.table_io.read_parquet(pred_art.path).collect()
        y_true = df["y_true"].to_numpy()
        y_proba = df["proba"].to_numpy()
        y_pred = (y_proba >= context.config.fusion.threshold).astype(int)
        
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_proba))
        }
        
        report_path = os.path.join(output_dir, "run_report.json")
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=4)
            
        context.event_bus.publish(Event(
            type="RUN_FINISHED",
            payload={
                "status": "ok",
                "total_time_s": time.time() - start_ts, # This is just for this task, but runner handles global
                "metrics": metrics,
                "report_path": report_path
            },
            ts=time.time(),
            run_id=context.run_id
        ))
        
        context.logger.info("validating", f"Evaluation complete: F1={metrics['f1']:.4f}", 
                            metrics=metrics, report_path=report_path)
        
        monitor.snapshot(self.name)
        return TaskResult(
            task_name=self.name,
            status="ok",
            duration_s=time.time() - start_ts,
            outputs=[report_path],
            meta=metrics
        )
