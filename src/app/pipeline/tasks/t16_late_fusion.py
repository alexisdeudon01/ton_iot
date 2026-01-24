import os
import time
import joblib
import numpy as np
import polars as pl
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.core.contracts.artifacts import PredictionArtifact
from src.app.pipeline.registry import TaskRegistry
from src.infra.models.sklearn_models import SklearnModel
from src.infra.models.torch_cnn import TorchCNNModel
from src.infra.models.tabnet_model import TabNetModel

@TaskRegistry.register("T16_LateFusion")
class T16_LateFusion(Task):
    def run(self, context: DAGContext) -> TaskResult:
        from src.infra.resources.monitor import ResourceMonitor
        monitor = ResourceMonitor(context.event_bus, context.run_id)
        monitor.snapshot(self.name)
        
        start_ts = time.time()
        output_path = os.path.join(context.config.paths.work_dir, "data", "predictions_fused.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        context.logger.info("predicting", "Performing Late Fusion (Averaging model probabilities)")

        cfg = context.config
        # Load splits for CIC and TON
        cic_splits_path = os.path.join(cfg.paths.work_dir, "data", "cic_splits.parquet")
        ton_splits_path = os.path.join(cfg.paths.work_dir, "data", "ton_splits.parquet")
        df_cic = context.table_io.read_parquet(cic_splits_path).collect()
        df_ton = context.table_io.read_parquet(ton_splits_path).collect()

        # Load preprocessors
        prep_cic = context.artifact_store.load_preprocess("preprocess_cic")
        prep_ton = context.artifact_store.load_preprocess("preprocess_ton")
        ct_cic = joblib.load(prep_cic.preprocess_path)
        ct_ton = joblib.load(prep_ton.preprocess_path)

        feature_order = context.artifact_store.load_table("cic_projected").feature_order

        def _load_model(model_type: str, dataset: str):
            model_art = context.artifact_store.load_model(f"model_{dataset}_{model_type}")
            if model_type in ["LR", "DT", "RF"]:
                model = SklearnModel(model_type, model_art.feature_order)
            elif model_type == "CNN":
                model = TorchCNNModel(model_art.feature_order)
            elif model_type == "TabNet":
                model = TabNetModel(model_art.feature_order)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            model.load(model_art.model_path)
            return model

        def _positive_proba(probas: np.ndarray) -> np.ndarray:
            if probas.ndim == 1:
                return probas
            if probas.shape[1] == 1:
                return probas[:, 0]
            return probas[:, 1]

        def _fuse_for_dataset(df: pl.DataFrame, dataset_name: str):
            X = df.select(feature_order).to_pandas()
            X_cic = ct_cic.transform(X)
            X_ton = ct_ton.transform(X)
            per_algo_preds = []
            per_algo_probas = []
            used_algos = []
            debug_snapshot = None

            for model_type in cfg.training.algorithms:
                try:
                    model_cic = _load_model(model_type, "cic")
                    model_ton = _load_model(model_type, "ton")
                except Exception as exc:
                    context.logger.warning("predicting", f"Skipping {model_type} in fusion: {exc}")
                    continue

                proba_cic = _positive_proba(model_cic.predict_proba(X_cic))
                proba_ton = _positive_proba(model_ton.predict_proba(X_ton))
                proba_fused = (proba_cic + proba_ton) / 2.0

                if debug_snapshot is None and proba_fused.shape[0] > 0:
                    debug_snapshot = {
                        "algo": model_type,
                        "proba_cic": proba_cic[:3].tolist(),
                        "proba_ton": proba_ton[:3].tolist(),
                        "proba_fused": proba_fused[:3].tolist(),
                    }

                per_algo_probas.append(proba_fused)
                used_algos.append(model_type)

                per_algo_preds.append(
                    pl.DataFrame({
                        "sample_id": df["sample_id"],
                        "proba": proba_fused,
                        "y_true": df["y"],
                        "dataset": dataset_name,
                        "model": model_type,
                        "split": df["split"],
                        "source_file": df["source_file"],
                    }).with_columns(pl.col("proba").cast(pl.Float64))
                )

            if not per_algo_probas:
                return pl.DataFrame(), pl.DataFrame()

            proba_global = np.mean(np.vstack(per_algo_probas), axis=0)

            if debug_snapshot:
                context.logger.info(
                    "predicting",
                    "Fusion sample check",
                    dataset=dataset_name,
                    algo=debug_snapshot["algo"],
                    proba_cic=debug_snapshot["proba_cic"],
                    proba_ton=debug_snapshot["proba_ton"],
                    proba_fused=debug_snapshot["proba_fused"],
                )

            global_df = pl.DataFrame({
                "sample_id": df["sample_id"],
                "proba": proba_global,
                "y_true": df["y"],
                "dataset": dataset_name,
                "model": "fused_global",
                "split": df["split"],
                "source_file": df["source_file"],
            }).with_columns(pl.col("proba").cast(pl.Float64))

            return pl.concat(per_algo_preds), global_df

        fused_algo_cic, fused_global_cic = _fuse_for_dataset(df_cic, "cic")
        fused_algo_ton, fused_global_ton = _fuse_for_dataset(df_ton, "ton")

        fused_by_algo = pl.concat([fused_algo_cic, fused_algo_ton]) if fused_algo_cic.height + fused_algo_ton.height > 0 else pl.DataFrame()
        fused_df = pl.concat([fused_global_cic, fused_global_ton]) if fused_global_cic.height + fused_global_ton.height > 0 else pl.DataFrame()

        # Save fused outputs
        context.table_io.write_parquet(fused_df, output_path)
        context.table_io.write_csv(fused_df, output_path.replace(".parquet", ".csv"))

        if fused_by_algo.height > 0:
            by_algo_path = output_path.replace("predictions_fused.parquet", "predictions_fused_by_algo.parquet")
            context.table_io.write_parquet(fused_by_algo, by_algo_path)
            context.table_io.write_csv(fused_by_algo, by_algo_path.replace(".parquet", ".csv"))
            context.logger.info("predicting", f"Fused-by-algo predictions saved to {by_algo_path}")
        
        artifact = PredictionArtifact(
            artifact_id="predictions_fused",
            path=output_path,
            version="1.0.0"
        )
        context.artifact_store.save_prediction(artifact)
        
        context.logger.info("predicting", f"Fused predictions saved to {output_path}", 
                            artifact=artifact.model_dump())
        
        monitor.snapshot(self.name)
        return TaskResult(
            task_name=self.name,
            status="ok",
            duration_s=time.time() - start_ts,
            outputs=["predictions_fused"]
        )
