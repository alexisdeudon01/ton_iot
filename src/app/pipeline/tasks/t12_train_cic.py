import os
import time
import joblib
import polars as pl
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.core.contracts.artifacts import ModelArtifact
from src.app.pipeline.registry import TaskRegistry
from src.infra.models.sklearn_models import SklearnModel

@TaskRegistry.register("T12_TrainCIC")
class T12_TrainCIC(Task):
    def run(self, context: DAGContext) -> TaskResult:
        start_ts = time.time()
        cic_art = context.artifact_store.load_table("cic_projected")
        prep_art = context.artifact_store.load_preprocess("preprocess_cic")
        
        # For demo, we train one model (RF)
        model_type = "RF"
        output_path = os.path.join(context.config.paths.work_dir, "models", f"cic_{model_type}.joblib")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        context.logger.info("training", f"Training {model_type} on CIC")
        
        df = context.table_io.read_parquet(cic_art.path).collect()
        X = df.select(cic_art.feature_order).to_pandas()
        y = df["y"].to_numpy()
        
        # Preprocess
        ct = joblib.load(prep_art.preprocess_path)
        X_transformed = ct.transform(X)
        
        model = SklearnModel(model_type, cic_art.feature_order, n_estimators=50, n_jobs=-1)
        model.train(X_transformed, y)
        model.save(output_path)
        
        artifact = ModelArtifact(
            artifact_id=f"model_cic_{model_type}",
            model_path=output_path,
            model_type=model_type,
            dataset="cic",
            feature_order=cic_art.feature_order,
            calibration="none",
            metrics_cv={},
            version="1.0.0"
        )
        context.artifact_store.save_model(artifact)
        
        return TaskResult(
            task_name=self.name,
            status="ok",
            duration_s=time.time() - start_ts,
            outputs=[artifact.artifact_id]
        )
