import os
import time
import polars as pl
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.core.contracts.artifacts import PredictionArtifact
from src.app.pipeline.registry import TaskRegistry

@TaskRegistry.register("T16_LateFusion")
class T16_LateFusion(Task):
    def run(self, context: DAGContext) -> TaskResult:
        from src.infra.resources.monitor import ResourceMonitor
        monitor = ResourceMonitor(context.event_bus, context.run_id)
        monitor.snapshot(self.name)
        
        start_ts = time.time()
        pred_cic_art = context.artifact_store.load_prediction("predictions_cic")
        pred_ton_art = context.artifact_store.load_prediction("predictions_ton")
        
        output_path = os.path.join(context.config.paths.work_dir, "data", "predictions_fused.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        context.logger.info("predicting", "Performing Late Fusion (Averaging model probabilities)")
        
        df_cic = context.table_io.read_parquet(pred_cic_art.path).collect()
        df_ton = context.table_io.read_parquet(pred_ton_art.path).collect()
        
        # Late Fusion: Average probabilities of all models for each sample
        # We must preserve 'split' to allow T17 to filter on 'test'
        fused_cic = df_cic.group_by(["dataset", "sample_id"]).agg([
            pl.col("proba").mean(),
            pl.col("y_true").first(),
            pl.col("split").first(),
            pl.col("source_file").first()
        ])
        
        fused_ton = df_ton.group_by(["dataset", "sample_id"]).agg([
            pl.col("proba").mean(),
            pl.col("y_true").first(),
            pl.col("split").first(),
            pl.col("source_file").first()
        ])
        
        fused_df = pl.concat([fused_cic, fused_ton])
        
        context.table_io.write_parquet(fused_df, output_path)
        context.table_io.write_csv(fused_df, output_path.replace(".parquet", ".csv"))
        
        artifact = PredictionArtifact(
            artifact_id="predictions_fused",
            path=output_path,
            version="1.0.0"
        )
        context.artifact_store.save_prediction(artifact)
        
        context.logger.info("predicting", f"Fused predictions saved to {output_path}")
        
        monitor.snapshot(self.name)
        return TaskResult(
            task_name=self.name,
            status="ok",
            duration_s=time.time() - start_ts,
            outputs=["predictions_fused"]
        )
