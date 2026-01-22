import os
import time
import joblib
import polars as pl
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.core.contracts.artifacts import PredictionArtifact
from src.app.pipeline.registry import TaskRegistry
from src.infra.models.sklearn_models import SklearnModel

@TaskRegistry.register("T15_PredictTON")
class T15_PredictTON(Task):
    def run(self, context: DAGContext) -> TaskResult:
        from src.infra.resources.monitor import ResourceMonitor
        monitor = ResourceMonitor(context.event_bus, context.run_id)
        monitor.snapshot(self.name)
        
        start_ts = time.time()
        ton_art = context.artifact_store.load_table("ton_projected")
        model_art = context.artifact_store.load_model("model_ton_RF")
        prep_art = context.artifact_store.load_preprocess("preprocess_ton")
        
        output_path = os.path.join(context.config.paths.work_dir, "data", "predictions_ton.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        context.logger.info("predicting", "Generating predictions for TON")
        
        df = context.table_io.read_parquet(ton_art.path).collect()
        X = df.select(ton_art.feature_order).to_pandas()
        
        ct = joblib.load(prep_art.preprocess_path)
        X_transformed = ct.transform(X)
        
        model = SklearnModel(model_art.model_type, model_art.feature_order)
        model.load(model_art.model_path)
        
        probas_raw = model.predict_proba(X_transformed)
        if probas_raw.shape[1] > 1:
            probas = probas_raw[:, 1]
        else:
            # Single class case
            if hasattr(model.model, "classes_") and model.model.classes_[0] == 1:
                probas = probas_raw[:, 0]
            else:
                probas = probas_raw[:, 0] * 0.0
        
        pred_df = pl.DataFrame({
            "proba": probas,
            "y_true": df["y"],
            "dataset": "ton",
            "model": model_art.model_type,
            "split": "test",
            "source_file": df["source_file"]
        })
        
        context.table_io.write_parquet(pred_df, output_path)
        
        artifact = PredictionArtifact(
            artifact_id="predictions_ton",
            path=output_path,
            version="1.0.0"
        )
        context.artifact_store.save_prediction(artifact)
        
        context.logger.info("predicting", f"Predictions TON saved to {output_path}")
        
        monitor.snapshot(self.name)
        return TaskResult(
            task_name=self.name,
            status="ok",
            duration_s=time.time() - start_ts,
            outputs=["predictions_ton"]
        )
