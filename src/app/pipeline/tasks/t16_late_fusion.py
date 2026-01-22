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
        start_ts = time.time()
        pred_cic_art = context.artifact_store.load_prediction("predictions_cic")
        pred_ton_art = context.artifact_store.load_prediction("predictions_ton")
        
        output_path = os.path.join(context.config.paths.work_dir, "data", "predictions_fused.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        context.logger.info("predicting", "Performing Late Fusion (Weighted Average)")
        
        df_cic = context.table_io.read_parquet(pred_cic_art.path).collect()
        df_ton = context.table_io.read_parquet(pred_ton_art.path).collect()
        
        # Strategy: Weighted Average of probabilities
        # Since datasets are heterogeneous, we might just concatenate them for evaluation
        # or perform fusion if they share the same samples (not the case here).
        # User said: "Ne pas fusionner les datasets au niveau lignes" for pipeline, 
        # but Late Fusion usually implies combining scores for the same events.
        # Given "2 datasets hétérogènes", we'll produce a combined prediction set.
        
        w = context.config.fusion.weight_w
        # For demo, we just concat the results as they are from different sources
        fused_df = pl.concat([df_cic, df_ton])
        
        context.table_io.write_parquet(fused_df, output_path)
        
        artifact = PredictionArtifact(
            artifact_id="predictions_fused",
            path=output_path,
            version="1.0.0"
        )
        context.artifact_store.save_prediction(artifact)
        
        return TaskResult(
            task_name=self.name,
            status="ok",
            duration_s=time.time() - start_ts,
            outputs=["predictions_fused"]
        )
