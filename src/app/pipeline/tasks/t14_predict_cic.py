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
from src.infra.models.torch_cnn import TorchCNNModel
from src.infra.models.tabnet_model import TabNetModel

@TaskRegistry.register("T14_PredictCIC")
class T14_PredictCIC(Task):
    def run(self, context: DAGContext) -> TaskResult:
        from src.infra.resources.monitor import ResourceMonitor
        monitor = ResourceMonitor(context.event_bus, context.run_id)
        monitor.snapshot(self.name)
        
        start_ts = time.time()
        cfg = context.config
        prep_art = context.artifact_store.load_preprocess("preprocess_cic")
        
        # Load splits instead of projected table
        splits_path = os.path.join(cfg.paths.work_dir, "data", "cic_splits.parquet")
        df = context.table_io.read_parquet(splits_path).collect()
        
        # We predict on the whole split set to have full results, 
        # but evaluation will filter on split=='test'
        X = df.select(context.artifact_store.load_table("cic_projected").feature_order).to_pandas()
        ct = joblib.load(prep_art.preprocess_path)
        X_transformed = ct.transform(X)

        algos = cfg.training.algorithms
        all_preds = []

        for model_type in algos:
            context.logger.info("predicting", f"Generating predictions for CIC with {model_type}")
            model_art = context.artifact_store.load_model(f"model_cic_{model_type}")
            
            if model_type in ["LR", "DT", "RF"]:
                model = SklearnModel(model_type, model_art.feature_order)
            elif model_type == "CNN":
                model = TorchCNNModel(model_art.feature_order)
            elif model_type == "TabNet":
                model = TabNetModel(model_art.feature_order)
                
            model.load(model_art.model_path)
            
            probas_raw = model.predict_proba(X_transformed)
            if probas_raw.shape[1] > 1:
                probas = probas_raw[:, 1]
            else:
                if hasattr(model.model, "classes_") and model.model.classes_[0] == 1:
                    probas = probas_raw[:, 0]
                else:
                    probas = probas_raw[:, 0] * 0.0
            
            pred_df = pl.DataFrame({
                "sample_id": df["sample_id"],
                "proba": probas,
                "y_true": df["y"],
                "dataset": "cic",
                "model": model_type,
                "split": df["split"],
                "source_file": df["source_file"]
            }).with_columns(pl.col("proba").cast(pl.Float64))
            all_preds.append(pred_df)

        full_pred_df = pl.concat(all_preds)
        output_path = os.path.join(cfg.paths.work_dir, "data", "predictions_cic.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        context.table_io.write_parquet(full_pred_df, output_path)
        context.table_io.write_csv(full_pred_df, output_path.replace(".parquet", ".csv"))
        
        artifact = PredictionArtifact(
            artifact_id="predictions_cic",
            path=output_path,
            version="1.0.0"
        )
        context.artifact_store.save_prediction(artifact)
        
        context.logger.info("predicting", f"Predictions CIC saved to {output_path}", 
                            artifact=artifact.model_dump())
        
        monitor.snapshot(self.name)
        return TaskResult(
            task_name=self.name,
            status="ok",
            duration_s=time.time() - start_ts,
            outputs=["predictions_cic"]
        )
