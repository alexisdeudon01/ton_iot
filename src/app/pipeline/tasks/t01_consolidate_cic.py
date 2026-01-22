import os
import glob
import time
import polars as pl
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.core.contracts.artifacts import TableArtifact
from src.app.pipeline.registry import TaskRegistry

@TaskRegistry.register("T01_ConsolidateCIC")
class T01_ConsolidateCIC(Task):
    def run(self, context: DAGContext) -> TaskResult:
        from src.infra.resources.monitor import ResourceMonitor
        monitor = ResourceMonitor(context.event_bus, context.run_id)
        monitor.snapshot(self.name)
        
        start_ts = time.time()
        cic_dir = context.config.paths.cic_dir_path
        output_path = os.path.join(context.config.paths.work_dir, "data", "cic_consolidated.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        context.logger.info("loading", f"Scanning CSVs in {cic_dir}")
        csv_files = glob.glob(os.path.join(cic_dir, "**", "*.csv"), recursive=True)
        
        if not csv_files:
            # Fallback for test or specific env
            csv_files = glob.glob(os.path.join("datasets/cic_ddos2019", "**", "*.csv"), recursive=True)
            
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {cic_dir}")

        lazy_frames = []
        # In test mode, only process first file to be extremely fast
        files_to_process = csv_files[:1] if context.config.test_mode else csv_files
        
        for f in files_to_process:
            context.logger.info("loading", f"Processing {f}")
            # Use n_rows in scan_csv for maximum speed in test mode
            scan_kwargs = {"n_rows": 100} if context.config.test_mode else {}
            lf = pl.scan_csv(f, infer_schema_length=100, ignore_errors=True, **scan_kwargs)
            
            # Clean column names (strip whitespace)
            lf = lf.rename({c: c.strip() for c in lf.collect_schema().names()})
            
            # Drop Unnamed: 0 if present
            cols = lf.collect_schema().names()
            if "Unnamed: 0" in cols:
                lf = lf.drop("Unnamed: 0")
            
            # Label rule: y=0 if Label == "BENIGN" else 1
            lf = lf.with_columns([
                pl.when(pl.col("Label") == "BENIGN").then(0).otherwise(1).alias("y"),
                pl.lit(f).alias("source_file")
            ])
            
            if context.config.test_mode:
                lf = lf.head(100) # Limit rows per file in test mode
                
            lazy_frames.append(lf)

        context.logger.info("cleaning", f"Concatenating {len(lazy_frames)} CIC datasets")
        # Use diagonal concat as columns might differ slightly between files
        full_lf = pl.concat(lazy_frames, how="diagonal")
        
        # Compute and log stats before collect
        context.logger.info("cleaning", "Collecting LazyFrame...")
        df = full_lf.collect()
        
        context.logger.info("writing", f"Writing Parquet to {output_path}", 
                            n_rows=df.height, n_cols=df.width, 
                            dtypes={c: str(t) for c, t in zip(df.columns, df.dtypes)})
        
        context.table_io.write_parquet(df, output_path, compression=context.config.io.parquet_compression)

        artifact = TableArtifact(
            artifact_id="cic_consolidated",
            name="CIC Consolidated",
            path=output_path,
            format="parquet",
            n_rows=df.height,
            n_cols=df.width,
            columns=df.columns,
            dtypes={col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
            version="1.0.0",
            source_step=self.name,
            fingerprint=str(hash(output_path)),
            stats={"csv_count": len(csv_files)}
        )
        context.artifact_store.save_table(artifact)
        
        context.logger.info("writing", f"Saved CIC consolidated: {df.height} rows, {df.width} cols", 
                            n_rows=df.height, n_cols=df.width)

        monitor.snapshot(self.name)
        return TaskResult(
            task_name=self.name,
            status="ok",
            duration_s=time.time() - start_ts,
            outputs=["cic_consolidated"]
        )
