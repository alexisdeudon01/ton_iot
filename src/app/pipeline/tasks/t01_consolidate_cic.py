import os
import time
import math
import polars as pl
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.core.contracts.artifacts import TableArtifact
from src.app.pipeline.registry import TaskRegistry

@TaskRegistry.register("T01_ConsolidateCIC")
class T01_ConsolidateCIC(Task):
    @staticmethod
    def _stratified_sample(df: pl.DataFrame, label_col: str, target_rows: int, seed: int) -> pl.DataFrame:
        if label_col not in df.columns or df.height == 0:
            return df
        target_rows = max(1, min(target_rows, df.height))
        fraction = min(1.0, target_rows / df.height)

        def _sample_group(group: pl.DataFrame) -> pl.DataFrame:
            if fraction >= 1.0:
                return group
            n = max(1, int(math.ceil(group.height * fraction)))
            if n >= group.height:
                return group
            return group.sample(n=n, seed=seed)

        return df.group_by(label_col).map_groups(_sample_group)

    def run(self, context: DAGContext) -> TaskResult:
        from src.infra.resources.monitor import ResourceMonitor
        monitor = ResourceMonitor(context.event_bus, context.run_id)
        monitor.snapshot(self.name)
        
        start_ts = time.time()
        cfg = context.config
        cic_dir = cfg.paths.cic_dir_path
        output_path = os.path.join(cfg.paths.work_dir, "data", "cic_consolidated.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        context.logger.info("loading", f"Consolidating CIC datasets from {cic_dir}")
        
        # List all CSV files
        if not os.path.exists(cic_dir):
            return TaskResult(task_name=self.name, status="failed", duration_s=time.time() - start_ts, error=f"CIC directory not found: {cic_dir}")
            
        csv_files = []
        for root, dirs, files in os.walk(cic_dir):
            for f in files:
                if f.endswith(".csv"):
                    csv_files.append(os.path.join(root, f))
        
        if not csv_files:
            context.logger.error("loading", "ERROR: No CSV files found for CIC-DDoS2019. Aborting.")
            return TaskResult(task_name=self.name, status="failed", duration_s=time.time() - start_ts, error="No CIC CSV files found")

        if cfg.test_mode:
            csv_files = csv_files[:cfg.test_max_files_per_dataset]

        dfs = []
        for f in csv_files:
            # Eager read to stabilize schema per file
            try:
                # Clean column names (trim spaces)
                df_temp = pl.read_csv(
                    f, 
                    infer_schema_length=1000, 
                    null_values=["Infinity", "NaN", "nan", "inf", "None"],
                    ignore_errors=True,
                    n_rows=cfg.test_row_limit_per_file if cfg.test_mode else None
                )
                
                # Normalize column names
                df_temp.columns = [c.strip() for c in df_temp.columns]
                
                # Aggressive type unification to avoid Float64/String conflicts
                # Columns that must be strings
                string_cols = {"Label", "Timestamp", "source_file", "Flow ID", "Source IP", "Destination IP"}
                
                # All other columns are forced to Float64
                # Non-convertible values become null (e.g., \"Infinity\")
                cast_exprs = []
                for col in df_temp.columns:
                    if col in string_cols:
                        cast_exprs.append(pl.col(col).cast(pl.String))
                    else:
                        cast_exprs.append(pl.col(col).cast(pl.Float64, strict=False))
                
                df_temp = df_temp.with_columns(cast_exprs)
                df_temp = df_temp.with_columns(pl.lit(os.path.basename(f)).alias("source_file"))
                dfs.append(df_temp)
                context.logger.info("loading", f"  [OK] {os.path.basename(f)} loaded ({df_temp.height} rows)")
            except Exception as e:
                context.logger.warning("loading", f"  [SKIP] Error on {f}: {e}")

        if not dfs:
            return TaskResult(task_name=self.name, status="failed", duration_s=time.time() - start_ts, error="No valid CIC CSV files loaded")

        # Eager diagonal concatenation
        df_full = pl.concat(dfs, how="diagonal")

        df_full = df_full.filter(pl.col("Label").is_not_null())

        # --- INTERACTIVE VALIDATION ---
        validation_size = getattr(cfg, "validation_sample_size", 10000)
        df_val = self._stratified_sample(df_full, "Label", validation_size, cfg.seed)

        print("\n--- CIC-DDoS2019 DATA VALIDATION ---")
        print(f"Files found: {len(csv_files)}")
        print(f"Number of columns (features): {df_full.width}")
        print(f"Number of rows (sample/total): {df_val.height} / {df_full.height}")
        
        # Label distribution preview
        if "Label" in df_val.columns:
            counts = df_val["Label"].value_counts().to_dicts()
            print("Proposed label distribution:")
            for r in counts:
                label = r.get("Label") or r.get("y")
                print(f"  - {label} : {r['count']} rows")
        
        print(f"\nFeature list: {df_full.columns[:20]} ... (+{max(0, len(df_full.columns)-20)})")
        
        print("\nRandom sample row:")
        print(df_val.sample(1).to_pandas().iloc[0].to_dict())
        
        try:
            confirm = input("\n?> Confirm use of these CIC data? (y/n): ").lower()
            if confirm != 'y':
                return TaskResult(task_name=self.name, status="failed", duration_s=time.time() - start_ts, error="User validation refused for CIC")
        except EOFError:
            pass  # Non-interactive mode

        # --- STRATIFIED SAMPLING ---
        sampling_ratio = cfg.sample_ratio
        context.logger.info("sampling", f"Performing stratified sampling at {sampling_ratio*100}%")

        target_rows = max(1, int(math.ceil(df_full.height * sampling_ratio)))
        df = self._stratified_sample(df_full, "Label", target_rows, cfg.seed)

        context.logger.info("sampling", f"Sampling complete: {df_full.height} -> {df.height} rows")

        # Stratification validation plot
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            strat_dir = os.path.join("graph", "stratification")
            os.makedirs(strat_dir, exist_ok=True)

            def _dist(frame: pl.DataFrame, col: str) -> pd.Series:
                counts = frame.select(col).to_pandas()[col].value_counts(normalize=True)
                return counts.sort_index()

            full_dist = _dist(df_full, "Label")
            sample_dist = _dist(df, "Label")
            all_labels = sorted(set(full_dist.index).union(sample_dist.index), key=lambda x: str(x))
            label_names = [str(l) for l in all_labels]
            full_vals = [float(full_dist.get(l, 0.0)) for l in all_labels]
            sample_vals = [float(sample_dist.get(l, 0.0)) for l in all_labels]

            x = range(len(all_labels))
            plt.figure(figsize=(10, 5))
            plt.bar([i - 0.2 for i in x], full_vals, width=0.4, label="Full")
            plt.bar([i + 0.2 for i in x], sample_vals, width=0.4, label="Sample")
            plt.xticks(list(x), label_names, rotation=45, ha="right")
            plt.ylabel("Proportion")
            plt.title("Stratification check (CIC)")
            plt.legend()
            plt.tight_layout()
            out_path = os.path.join(strat_dir, "cic_stratification.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            context.logger.info("sampling", f"Stratification chart saved: {out_path}")
        except Exception as e:
            context.logger.warning("sampling", f"Stratification chart failed: {e}")
        
        # Add global sample_id
        df = df.with_row_index("sample_id")
        
        # Basic cleaning: rename 'Label' to 'y' if needed or create 'y'
        if "Label" in df.columns:
            df = df.with_columns([
                pl.when(pl.col("Label") == "BENIGN").then(0).otherwise(1).alias("y")
            ])
        
        y_counts = df.group_by("y").count().to_dicts()
        label_balance = {str(r["y"]): int(r["count"]) for r in y_counts}

        context.table_io.write_parquet(df, output_path)
        
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
        
        context.logger.info("writing", f"CIC Consolidation complete. Output: {output_path}", 
                            artifact=artifact.model_dump())
        
        monitor.snapshot(self.name)
        return TaskResult(task_name=self.name, status="ok", duration_s=time.time() - start_ts, outputs=["cic_consolidated"])
