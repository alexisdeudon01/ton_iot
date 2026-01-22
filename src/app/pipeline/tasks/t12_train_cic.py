import os
import time
import joblib
import polars as pl
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.core.contracts.artifacts import ModelArtifact
from src.app.pipeline.registry import TaskRegistry

@TaskRegistry.register("T12_TrainCIC")
class T12_TrainCIC(Task):
    def run(self, context: DAGContext) -> TaskResult:
        from src.infra.resources.monitor import ResourceMonitor
        from src.infra.models.sklearn_models import SklearnModel
        from src.infra.models.torch_cnn import TorchCNNModel
        from src.infra.models.tabnet_model import TabNetModel
        
        monitor = ResourceMonitor(context.event_bus, context.run_id)
        monitor.snapshot(self.name)
        
        start_ts = time.time()
        cic_art = context.artifact_store.load_table("cic_projected")
        prep_art = context.artifact_store.load_preprocess("preprocess_cic")
        
        df = context.table_io.read_parquet(cic_art.path).collect()
        X = df.select(cic_art.feature_order).to_pandas()
        y = df["y"].to_numpy()
        
        ct = joblib.load(prep_art.preprocess_path)
        X_transformed = ct.transform(X)
        
        algos = context.config.training.algorithms
        outputs = []
        per_algo_perf = {}

        for model_type in algos:
            algo_start = time.time()
            context.logger.info("training", f"Training {model_type} on CIC")
            output_path = os.path.join(context.config.paths.work_dir, "models", f"cic_{model_type}.model")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if model_type in ["LR", "DT", "RF"]:
                model = SklearnModel(model_type, cic_art.feature_order)
            elif model_type == "CNN":
                model = TorchCNNModel(cic_art.feature_order, epochs=2 if context.config.test_mode else 10)
            elif model_type == "TabNet":
                model = TabNetModel(cic_art.feature_order)
            
            try:
                model.train(X_transformed, y)
                model.save(output_path)
                duration = time.time() - algo_start
                per_algo_perf[model_type] = {"train_duration_s": duration}
                
                artifact = ModelArtifact(
                    artifact_id=f"model_cic_{model_type}",
                    model_path=output_path,
                    model_type=model_type,
                    dataset="cic",
                    feature_order=cic_art.feature_order,
                    calibration="none",
                    metrics_cv={"train_duration_s": duration},
                    version="1.0.0"
                )
                context.artifact_store.save_model(artifact)
                outputs.append(artifact.artifact_id)
                context.logger.info("training", f"Model CIC {model_type} saved in {duration:.2f}s")
            except Exception as e:
                context.logger.warning("training", f"Failed to train {model_type}: {e}")
                continue

        monitor.snapshot(self.name)
        return TaskResult(
            task_name=self.name,
            status="ok",
            duration_s=time.time() - start_ts,
            outputs=outputs,
            meta=per_algo_perf
        )
