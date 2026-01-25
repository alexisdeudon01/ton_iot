import os
import time
import math
import polars as pl
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.core.contracts.artifacts import TableArtifact
from src.app.pipeline.registry import TaskRegistry

@TaskRegistry.register("T02_CleanTON")
class T02_CleanTON(Task):
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
        ton_path = cfg.paths.ton_csv_path
        output_path = os.path.join(cfg.paths.work_dir, "data", "ton_clean.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        context.logger.info("loading", f"Reading TON dataset from {ton_path}")
        if not os.path.exists(ton_path):
            context.logger.error("loading", f"ERROR: ToN-IoT file not found: {ton_path}. Aborting.")
            return TaskResult(task_name=self.name, status="failed", duration_s=time.time() - start_ts, error=f"File not found: {ton_path}")

        lf = pl.scan_csv(ton_path, infer_schema_length=100)

        # Keep only type in {"normal", "ddos"}
        lf = lf.filter(pl.col("type").is_in(["normal", "ddos"]))
        lf = lf.with_columns([
            pl.when(pl.col("type") == "ddos").then(1).otherwise(0).alias("y"),
            pl.lit(ton_path).alias("source_file")
        ])
        
        # Add global sample_id
        lf = lf.with_row_index("sample_id")
        
        df_full = lf.collect()

        # --- INTERACTIVE VALIDATION ---
        validation_size = getattr(cfg, "validation_sample_size", 10000)
        df_val = self._stratified_sample(df_full, "type", validation_size, cfg.seed)

        print("\n--- ToN-IoT DATA VALIDATION ---")
        print(f"Source file: {ton_path}")
        print(f"Number of columns (features): {df_full.width}")
        print(f"Number of rows (sample/total): {df_val.height} / {df_full.height}")

        if "type" in df_val.columns:
            counts = df_val["type"].value_counts().to_dicts()
            print("Proposed label distribution (type):")
            for r in counts:
                print(f"  - {r['type']} : {r['count']} rows")

        print(f"\nFeature list: {df_full.columns[:20]} ... (+{max(0, len(df_full.columns)-20)})")

        print("\nRandom sample row:")
        print(df_val.sample(1).to_pandas().iloc[0].to_dict())

        try:
            confirm = input("\n?> Confirm use of these ToN-IoT data? (y/n): ").lower()
            if confirm != 'y':
                return TaskResult(task_name=self.name, status="failed", duration_s=time.time() - start_ts, error="User validation refused for ToN-IoT")
        except EOFError:
            pass  # Non-interactive mode

        # --- STRATIFIED SAMPLING ---
        sampling_ratio = cfg.sample_ratio
        context.logger.info("sampling", f"Performing stratified sampling at {sampling_ratio*100}%")
        
        # Stratification sur la colonne 'type'
        target_rows = max(1, int(math.ceil(df_full.height * sampling_ratio)))
        df = self._stratified_sample(df_full, "type", target_rows, cfg.seed)
        
        reason = f"stratified_sampling_{int(sampling_ratio*100)}"
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

            full_dist = _dist(df_full, "type")
            sample_dist = _dist(df, "type")
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
            plt.title("Stratification check (ToN-IoT)")
            plt.legend()
            plt.tight_layout()
            out_path = os.path.join(strat_dir, "ton_stratification.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            context.logger.info("sampling", f"Stratification chart saved: {out_path}")
        except Exception as e:
            context.logger.warning("sampling", f"Stratification chart failed: {e}")
        
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
        context.logger.info("writing", "TON Cleaning complete", 
                            artifact=artifact.model_dump())
        
        monitor.snapshot(self.name)
        return TaskResult(task_name=self.name, status="ok", duration_s=time.time() - start_ts, outputs=["ton_clean"])
