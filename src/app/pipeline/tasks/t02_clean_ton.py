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
        from src.infra.resources.monitor import ResourceMonitor
        monitor = ResourceMonitor(context.event_bus, context.run_id)
        monitor.snapshot(self.name)
        
        start_ts = time.time()
        cfg = context.config
        ton_path = cfg.paths.ton_csv_path
        output_path = os.path.join(cfg.paths.work_dir, "data", "ton_clean.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        context.logger.info("loading", f"Reading TON dataset from {ton_path}")
        lf = pl.scan_csv(ton_path, infer_schema_length=100)
        
        # Keep only type in {"normal", "ddos"}
        lf = lf.filter(pl.col("type").is_in(["normal", "ddos"]))
        lf = lf.with_columns([
            pl.when(pl.col("type") == "ddos").then(1).otherwise(0).alias("y"),
            pl.lit(ton_path).alias("source_file")
        ])
        
        # Add global sample_id
        lf = lf.with_row_index("sample_id")
        
        if cfg.test_mode:
            # Apply global cap
            lf = lf.head(cfg.test_max_rows_total_per_dataset)
            reason = "test_mode"
        else:
            reason = "full_run"
            
        df = lf.collect()
        
        # Robust balance extraction for Polars
        y_counts = df.group_by("y").count().to_dicts()
        label_balance = {str(r["y"]): int(r["count"]) for r in y_counts}
        
        if len(label_balance) < 2:
            context.logger.warning("cleaning", "Only one class detected in TON sample", balance=label_balance)

        # Logs obligatoires
        context.logger.info("writing", "TON Cleaning complete",
                            test_mode=cfg.test_mode,
                            sample_ratio=cfg.sample_ratio,
                            test_max_rows_total=cfg.test_max_rows_total_per_dataset,
                            n_rows_out=df.height,
                            n_cols_out=df.width,
                            label_balance=label_balance,
                            unique_source_files=1,
                            top_10_cols=df.columns[:10],
                            num_cols=len([c for c, t in zip(df.columns, df.dtypes) if t.is_numeric()]),
                            cat_cols=len([c for c, t in zip(df.columns, df.dtypes) if t == pl.String]),
                            reason_for_row_limit=reason)
        
        context.table_io.write_parquet(df, output_path)
        context.table_io.write_csv(df, output_path.replace(".parquet", ".csv"))

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
            stats={"label_balance": label_balance}
        )
        context.artifact_store.save_table(artifact)
        
        monitor.snapshot(self.name)
        return TaskResult(task_name=self.name, status="ok", duration_s=time.time() - start_ts, outputs=["ton_clean"])
