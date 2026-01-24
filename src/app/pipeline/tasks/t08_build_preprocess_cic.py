import os
import time
import joblib
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.core.contracts.artifacts import PreprocessArtifact
from src.app.pipeline.registry import TaskRegistry
from src.app.pipeline.universal_feature_mapping import OUTLIER_FEATURES, RATIO_FEATURES
from src.infra.preprocessing.log_winsorizer import LogWinsorizer

@TaskRegistry.register("T08_BuildPreprocessCIC")
class T08_BuildPreprocessCIC(Task):
    def run(self, context: DAGContext) -> TaskResult:
        from src.infra.resources.monitor import ResourceMonitor
        monitor = ResourceMonitor(context.event_bus, context.run_id)
        monitor.snapshot(self.name)
        
        start_ts = time.time()
        cic_art = context.artifact_store.load_table("cic_projected")
        f_common = cic_art.feature_order
        f_outlier = [f for f in OUTLIER_FEATURES if f in f_common]
        f_ratio = [f for f in RATIO_FEATURES if f in f_common]
        extra = [f for f in f_common if f not in f_outlier and f not in f_ratio]
        f_outlier.extend(extra)
        
        output_path = os.path.join(context.config.paths.work_dir, "artifacts", "preprocess_cic.joblib")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        context.logger.info("cleaning", f"Building ColumnTransformer for CIC ({len(f_common)} features)")
        
        # Simple scaler for all common features (assuming they are numeric after projection)
        # In a real scenario, we'd separate numeric and categorical
        outlier_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("logwinsor", LogWinsorizer()),
            ("scaler", RobustScaler()),
        ])
        ratio_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ])
        ct = ColumnTransformer([
            ("outlier", outlier_pipeline, f_outlier),
            ("ratio", ratio_pipeline, f_ratio),
        ])
        
        # Fit on a sample
        df_sample = context.table_io.read_parquet(cic_art.path).head(10000).collect()
        ct.fit(df_sample.to_pandas())
        
        joblib.dump(ct, output_path)
        
        artifact = PreprocessArtifact(
            artifact_id="preprocess_cic",
            preprocess_path=output_path,
            num_features=f_common,
            cat_features=[],
            feature_order=f_common,
            steps={
                "outlier": "SimpleImputer+LogWinsorizer+RobustScaler",
                "ratio": "SimpleImputer+RobustScaler",
            },
            version="1.0.0"
        )
        context.artifact_store.save_preprocess(artifact)
        
        context.logger.info("cleaning", f"Preprocess CIC saved to {output_path}")
        
        monitor.snapshot(self.name)
        return TaskResult(
            task_name=self.name,
            status="ok",
            duration_s=time.time() - start_ts,
            outputs=["preprocess_cic"]
        )
