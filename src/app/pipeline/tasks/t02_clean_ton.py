import os
import time
import polars as pl
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.core.contracts.artifacts import TableArtifact
from src.app.pipeline.registry import TaskRegistry

@TaskRegistry.register("T02_CleanTON")
class T02_CleanTON(Task):
    def run(self, context: DAGContext) -> TaskResult:
        start_ts = time.time()
        ton_path = context.config.paths.ton_csv_path
        output_path = os.path.join(context.config.paths.work_dir, "data", "ton_clean.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        context.logger.info("loading", f"Reading TON dataset from {ton_path}")
        
        # Use scan_csv for lazy loading
        lf = pl.scan_csv(ton_path, infer_schema_length=10000)
        
        # Keep only type in {"normal", "ddos"}
        # y=1 if type=="ddos" else 0
        lf = lf.filter(pl.col("type").is_in(["normal", "ddos"]))
        lf = lf.with_columns([
            pl.when(pl.col("type") == "ddos").then(1).otherwise(0).alias("y"),
            pl.lit(ton_path).alias("source_file")
        ])
        
        context.logger.info("cleaning", "Filtering TON for normal/ddos types and mapping labels")
        
        df = lf.collect()
        
        context.table_io.write_parquet(df, output_path, compression=context.config.io.parquet_compression)

        artifact = TableArtifact(
            artifact_id="ton_clean",
            name="TON Clean",
            path=output_path,
            format="parquet",
            n_rows=df.height,
            n_cols=df.width,
            columns=df.columns,
            dtypes={col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
            version="1.0.0",
            source_step=self.name,
            fingerprint=str(hash(output_path)),
            stats={"original_path": ton_path}
        )
        context.artifact_store.save_table(artifact)
        
        context.logger.info("writing", f"Saved TON clean: {df.height} rows, {df.width} cols", 
                            n_rows=df.height, n_cols=df.width)

        return TaskResult(
            task_name=self.name,
            status="ok",
            duration_s=time.time() - start_ts,
            outputs=["ton_clean"]
        )
