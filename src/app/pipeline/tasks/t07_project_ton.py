import os
import time
import polars as pl
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.core.contracts.artifacts import TableArtifact
from src.app.pipeline.registry import TaskRegistry

@TaskRegistry.register("T07_ProjectTON")
class T07_ProjectTON(Task):
    def run(self, context: DAGContext) -> TaskResult:
        start_ts = time.time()
        ton_art = context.artifact_store.load_table("ton_clean")
        align_art = context.artifact_store.load_alignment("alignment_spec")
        
        f_common = align_art.F_common
        output_path = os.path.join(context.config.paths.work_dir, "data", "ton_projected.parquet")
        
        context.logger.info("loading", f"Projecting TON to {len(f_common)} features")
        
        lf = context.table_io.read_parquet(ton_art.path)
        
        # Select common features + label + metadata
        cols_to_keep = f_common + ["y", "source_file"]
        lf_projected = lf.select(cols_to_keep)
        
        df = lf_projected.collect()
        context.table_io.write_parquet(df, output_path)
        
        artifact = TableArtifact(
            artifact_id="ton_projected",
            name="TON Projected",
            path=output_path,
            format="parquet",
            n_rows=df.height,
            n_cols=df.width,
            columns=df.columns,
            dtypes={col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
            feature_order=f_common,
            version="1.0.0",
            source_step=self.name,
            fingerprint=str(hash(output_path)),
            stats={}
        )
        context.artifact_store.save_table(artifact)
        
        return TaskResult(
            task_name=self.name,
            status="ok",
            duration_s=time.time() - start_ts,
            outputs=["ton_projected"]
        )
