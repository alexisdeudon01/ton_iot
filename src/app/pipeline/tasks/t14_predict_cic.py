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

@TaskRegistry.register("T14_PredictCIC")
class T14_PredictCIC(Task):
    def run(self, context: DAGContext) -> TaskResult:
        start_ts = time.time()
        cic_art = context.artifact_store.load_table("cic_projected")
        model_art = context.artifact_store.load_model("model_cic_RF")
        prep_art = context.artifact_store.load_preprocess("preprocess_cic")
        
        output_path = os.path.join(context.config.paths.work_dir, "data", "predictions_cic.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        context.logger.info("predicting", "Generating predictions for CIC")
        
        df = context.table_io.read_parquet(cic_art.path).collect()
        X = df.select(cic_art.feature_order).to_pandas()
        
        ct = joblib.load(prep_art.preprocess_path)
        X_transformed = ct.transform(X)
        
        model = SklearnModel(model_art.model_type, model_art.feature_order)
        model.load(model_art.model_path)
        
        probas = model.predict_proba(X_transformed)[:, 1]
        
        pred_df = pl.DataFrame({
            "proba": probas,
            "y_true": df["y"],
            "dataset": "cic",
            "model": model_art.model_type,
            "split": "test", # Simplified
            "source_file": df["source_file"]
        })
        
        context.table_io.write_parquet(pred_df, output_path)
        
        artifact = PredictionArtifact(
            artifact_id="predictions_cic",
            path=output_path,
            version="1.0.0"
        )
        context.artifact_store.save_prediction(artifact)
        
        return TaskResult(
            task_name=self.name,
            status="ok",
            duration_s=time.time() - start_ts,
            outputs=["predictions_cic"]
        )
