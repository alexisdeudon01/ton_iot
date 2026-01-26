import os
import json
import time
import yaml
import pandas as pd
import polars as pl

from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.app.pipeline.registry import TaskRegistry
from src.app.pipeline.universal_feature_mapping import UNIVERSAL_FEATURES
from mcdm.topsis_tppis_report import build_df_sources_from_run_report, generate_topsis_report


@TaskRegistry.register("T19_TopsisReport")
class T19_TopsisReport(Task):
    """
    Automated MCDA (AHP/TOPSIS) report generation with visual outputs.
    Saves all plots under /topsis_tppis at project root.
    """

    def run(self, context: DAGContext) -> TaskResult:
        start_ts = time.time()
        output_dir = os.path.join(os.getcwd(), "topsis_tppis")
        os.makedirs(output_dir, exist_ok=True)

        report_path = os.path.join("reports", "run_report.json")
        if not os.path.exists(report_path):
            return TaskResult(
                task_name=self.name,
                status="failed",
                duration_s=time.time() - start_ts,
                error="Evaluation report not found.",
            )

        with open(report_path, "r") as f:
            run_report = json.load(f)

        df_sources = build_df_sources_from_run_report(run_report)
        if df_sources.empty:
            return TaskResult(
                task_name=self.name,
                status="failed",
                duration_s=time.time() - start_ts,
                error="No fused alternatives available for TOPSIS report.",
            )

        # Load projected data for feature analysis
        cic_art = context.artifact_store.load_table("cic_projected")
        ton_art = context.artifact_store.load_table("ton_projected")
        df_cic = context.table_io.read_parquet(cic_art.path).collect().to_pandas()
        df_ton = context.table_io.read_parquet(ton_art.path).collect().to_pandas()
        df_features = pd.concat([df_cic, df_ton], ignore_index=True)

        if "y" in df_features.columns:
            y = df_features["y"].fillna(0).astype(int)
        else:
            y = pd.Series([0] * len(df_features))

        df_features = df_features[[c for c in UNIVERSAL_FEATURES if c in df_features.columns] + ["y"]]

        # Use pipeline config weights if available
        weights = [0.70, 0.15, 0.15]
        try:
            with open("config/pipeline.yaml", "r") as f:
                cfg = yaml.safe_load(f)
            pillars = cfg.get("mcdm_hierarchy", {}).get("pillars", [])
            if len(pillars) >= 3:
                weights = [pillars[0].get("weight", 0.70), pillars[1].get("weight", 0.15), pillars[2].get("weight", 0.15)]
        except Exception:
            pass

        outputs = generate_topsis_report(
            df_sources=df_sources,
            df_features=df_features,
            y=y,
            output_dir=output_dir,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            weights=weights,
        )

        # Write simple index
        index_path = os.path.join(output_dir, "README.md")
        with open(index_path, "w") as f:
            f.write("# TOPSIS Visual Report\n\n")
            f.write("Generated plots:\n")
            for p in outputs.get("plots", []):
                f.write(f"- {p}\n")

        # Update run_report metadata
        if "_metadata" in run_report:
            files = []
            for p in outputs.get("plots", []):
                files.append({
                    "name": os.path.basename(p),
                    "path": os.path.abspath(p),
                    "url": f"file://{os.path.abspath(p)}",
                    "type": "topsis_report",
                })
            files.append({
                "name": os.path.basename(index_path),
                "path": os.path.abspath(index_path),
                "url": f"file://{os.path.abspath(index_path)}",
                "type": "topsis_report",
            })
            run_report["_metadata"]["outputs"]["generated_files"].extend(files)
            with open(report_path, "w") as f:
                json.dump(run_report, f, indent=4)

        return TaskResult(
            task_name=self.name,
            status="ok",
            duration_s=time.time() - start_ts,
            outputs=[output_dir, index_path],
        )
