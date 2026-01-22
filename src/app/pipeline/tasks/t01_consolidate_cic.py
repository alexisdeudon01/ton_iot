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
        cfg = context.config
        cic_dir = cfg.paths.cic_dir_path
        output_path = os.path.join(cfg.paths.work_dir, "data", "cic_consolidated.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        csv_files = glob.glob(os.path.join(cic_dir, "**", "*.csv"), recursive=True)
        n_files_detected = len(csv_files)
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {cic_dir}")

        files_to_process = csv_files[:cfg.test_max_files_per_dataset] if cfg.test_mode else csv_files
        n_files_used = len(files_to_process)

        context.logger.info("loading", f"Processing CIC: {n_files_used}/{n_files_detected} files",
                            test_mode=cfg.test_mode,
                            test_max_files=cfg.test_max_files_per_dataset)

        lazy_frames = []
        for f in files_to_process:
            scan_kwargs = {}
            if cfg.test_mode:
                scan_kwargs["n_rows"] = cfg.test_row_limit_per_file
            
            lf = pl.scan_csv(f, infer_schema_length=100, ignore_errors=True, **scan_kwargs)
            
            # Clean column names
            lf = lf.rename({c: c.strip() for c in lf.collect_schema().names()})
            
            # Drop Unnamed: 0
            cols = lf.collect_schema().names()
            if "Unnamed: 0" in cols:
                lf = lf.drop("Unnamed: 0")
            
            # Label rule: y=0 if Label == "BENIGN" else 1
            lf = lf.with_columns([
                pl.when(pl.col("Label") == "BENIGN").then(0).otherwise(1).alias("y"),
                pl.lit(f).alias("source_file")
            ])
            
            # Cast problematic columns to String
            prob_cols = ["Flow Bytes/s", "Flow Packets/s", "SimillarHTTP"]
            existing = set(lf.collect_schema().names())
            lf = lf.with_columns([
                pl.col(c).cast(pl.String) for c in prob_cols if c in existing
            ])
            
            lazy_frames.append(lf)

        full_lf = pl.concat(lazy_frames, how="diagonal")
        full_lf = full_lf.with_row_index("sample_id")
        
        if cfg.test_mode:
            full_lf = full_lf.head(cfg.test_max_rows_total_per_dataset)
            reason = "test_mode"
        else:
            reason = "full_run"

        df = full_lf.collect()
        
        # Sanity check: label balance
        y_counts = df["y"].value_counts().to_dicts()
        label_balance = {str(r["y"]): r["count"] for r in y_counts}
        
        if len(label_balance) < 2:
            context.logger.warning("cleaning", "Only one class detected in CIC sample", balance=label_balance)

        # Logs obligatoires
        context.logger.info("writing", "CIC Consolidation complete",
                            n_files_detected=n_files_detected,
                            n_files_used=n_files_used,
                            test_mode=cfg.test_mode,
                            sample_ratio=cfg.sample_ratio,
                            test_row_limit_per_file=cfg.test_row_limit_per_file,
                            test_max_rows_total=cfg.test_max_rows_total_per_dataset,
                            n_rows_out=df.height,
                            n_cols_out=df.width,
                            label_balance=label_balance,
                            unique_source_files=df["source_file"].n_unique(),
                            top_10_cols=df.columns[:10],
                            num_cols=len([c for c, t in zip(df.columns, df.dtypes) if t.is_numeric()]),
                            cat_cols=len([c for c, t in zip(df.columns, df.dtypes) if t == pl.String]),
                            sampling_reason=reason)

        context.table_io.write_parquet(df, output_path)
        context.table_io.write_csv(df, output_path.replace(".parquet", ".csv"))

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
            stats={"label_balance": label_balance}
        )
        context.artifact_store.save_table(artifact)
        
        monitor.snapshot(self.name)
        return TaskResult(task_name=self.name, status="ok", duration_s=time.time() - start_ts, outputs=["cic_consolidated"])
