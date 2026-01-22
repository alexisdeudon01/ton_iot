import os
import time
import joblib
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.core.contracts.artifacts import PreprocessArtifact
from src.app.pipeline.registry import TaskRegistry

@TaskRegistry.register("T09_BuildPreprocessTON")
class T09_BuildPreprocessTON(Task):
    def run(self, context: DAGContext) -> TaskResult:
        start_ts = time.time()
        ton_art = context.artifact_store.load_table("ton_projected")
        f_common = ton_art.feature_order
        
        output_path = os.path.join(context.config.paths.work_dir, "artifacts", "preprocess_ton.joblib")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        context.logger.info("cleaning", f"Building ColumnTransformer for TON ({len(f_common)} features)")
        
        ct = ColumnTransformer([
            ("scaler", RobustScaler(), f_common)
        ])
        
        df_sample = context.table_io.read_parquet(ton_art.path).head(10000).collect()
        ct.fit(df_sample.to_pandas())
        
        joblib.dump(ct, output_path)
        
        artifact = PreprocessArtifact(
            artifact_id="preprocess_ton",
            preprocess_path=output_path,
            num_features=f_common,
            cat_features=[],
            feature_order=f_common,
            steps={"scaler": "RobustScaler"},
            version="1.0.0"
        )
        context.artifact_store.save_preprocess(artifact)
        
        return TaskResult(
            task_name=self.name,
            status="ok",
            duration_s=time.time() - start_ts,
            outputs=["preprocess_ton"]
        )
