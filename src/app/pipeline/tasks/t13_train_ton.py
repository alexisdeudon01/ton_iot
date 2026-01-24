import os
import time
import joblib
import polars as pl
import pandas as pd
import numpy as np
import psutil
import tracemalloc
from sklearn.model_selection import train_test_split
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.core.contracts.artifacts import ModelArtifact
from src.app.pipeline.registry import TaskRegistry

@TaskRegistry.register("T13_TrainTON")
class T13_TrainTON(Task):
    def run(self, context: DAGContext) -> TaskResult:
        from src.infra.resources.monitor import ResourceMonitor
        from src.infra.models.sklearn_models import SklearnModel
        from src.infra.models.torch_cnn import TorchCNNModel
        from src.infra.models.tabnet_model import TabNetModel, TABNET_AVAILABLE
        
        monitor = ResourceMonitor(context.event_bus, context.run_id)
        monitor.snapshot(self.name)
        
        start_ts = time.time()
        cfg = context.config
        ton_art = context.artifact_store.load_table("ton_projected")
        prep_art = context.artifact_store.load_preprocess("preprocess_ton")
        
        df = context.table_io.read_parquet(ton_art.path).collect()
        
        # Split Train/Val/Test (70/15/15)
        train_df, temp_df = train_test_split(
            df.to_pandas(), test_size=0.3, random_state=cfg.seed, stratify=df["y"]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=cfg.seed, stratify=temp_df["y"]
        )

        def get_balance(df_part):
            # df_part is pandas here
            counts = df_part["y"].value_counts().to_dict()
            return {str(k): int(v) for k, v in counts.items()}

        context.logger.info("training", "Data split complete",
                            train_count=len(train_df), train_balance=get_balance(train_df),
                            val_count=len(val_df), val_balance=get_balance(val_df),
                            test_count=len(test_df), test_balance=get_balance(test_df))

        # Save splits
        splits_path = os.path.join(cfg.paths.work_dir, "data", "ton_splits.parquet")
        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"
        full_split_df = pd.concat([train_df, val_df, test_df])
        context.table_io.write_parquet(pl.from_pandas(full_split_df), splits_path)

        X_train = train_df[ton_art.feature_order]
        y_train = train_df["y"].values
        
        ct = joblib.load(prep_art.preprocess_path)
        X_train_transformed = ct.transform(X_train)
        
        # Utilisation de la liste des algorithmes configurés
        algo_configs = {a.key: a.params for a in cfg.algorithms}
        outputs = []
        per_algo_perf = {}

        for model_type in cfg.training.algorithms:
            if model_type == "TabNet" and not TABNET_AVAILABLE:
                context.logger.warning("training", "TabNet not installed, skipping algorithm")
                continue

            algo_start = time.time()
            context.logger.info("training", f"Training {model_type} on TON", test_mode=cfg.test_mode)
            output_path = os.path.join(cfg.paths.work_dir, "models", f"ton_{model_type}.model")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Récupération des paramètres par défaut depuis la config
            kwargs = algo_configs.get(model_type, {}).copy()

            if model_type in ["LR", "DT", "RF"]:
                if cfg.test_mode and model_type == "RF":
                    kwargs.update({"n_estimators": 10, "max_depth": 5})
                model = SklearnModel(model_type, ton_art.feature_order, **kwargs)
            elif model_type == "CNN":
                epochs = 2 if cfg.test_mode else kwargs.get("epochs", 10)
                model = TorchCNNModel(ton_art.feature_order, epochs=epochs, batch_size=kwargs.get("batch_size", 128))
            elif model_type == "TabNet":
                model = TabNetModel(ton_art.feature_order, **kwargs)
            
            try:
                train_kwargs = {}
                if model_type == "TabNet":
                    train_kwargs["max_epochs"] = 5 if cfg.test_mode else 20

                process = psutil.Process()
                tracemalloc.start()
                cpu_start = process.cpu_percent(interval=None)
                train_start = time.time()
                model.train(X_train_transformed, y_train, **train_kwargs)
                duration = time.time() - train_start
                cpu_end = process.cpu_percent(interval=None)
                _, mem_peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                model.save(output_path)

                n_params = 0
                try:
                    if model_type == "LR":
                        n_params = model.model.coef_.size + model.model.intercept_.size
                    elif model_type == "DT":
                        n_params = model.model.tree_.node_count
                    elif model_type == "RF":
                        n_params = sum(t.tree_.node_count for t in model.model.estimators_)
                    elif model_type == "CNN":
                        n_params = sum(p.numel() for p in model.model.parameters())
                    elif model_type == "TabNet" and hasattr(model, "model") and model.model is not None:
                        n_params = sum(p.numel() for p in model.model.network.parameters())
                except Exception:
                    n_params = 0

                per_algo_perf[model_type] = {
                    "train_duration_s": duration,
                    "memory_bytes": int(mem_peak),
                    "cpu_percent": float(max(cpu_start, cpu_end)),
                    "n_params": int(n_params),
                }
                
                artifact = ModelArtifact(
                    artifact_id=f"model_ton_{model_type}",
                    model_path=output_path,
                    model_type=model_type,
                    dataset="ton",
                    feature_order=ton_art.feature_order,
                    calibration="none",
                    metrics_cv=per_algo_perf[model_type],
                    version="1.0.0"
                )
                context.artifact_store.save_model(artifact)
                outputs.append(artifact.artifact_id)
            except Exception as e:
                context.logger.warning("training", f"Failed to train {model_type}: {e}")
                continue

        monitor.snapshot(self.name)
        return TaskResult(task_name=self.name, status="ok", duration_s=time.time() - start_ts, outputs=outputs, meta=per_algo_perf)
