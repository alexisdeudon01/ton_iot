import os
import time
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.app.pipeline.registry import TaskRegistry


@TaskRegistry.register("T10_SamplingValidation")
class T10_SamplingValidation(Task):
    """Validate sampling quality with KS tests (validation vs sampled)."""

    def _ks_report(self, df_full: pd.DataFrame, df_sample: pd.DataFrame, name: str, output_dir: str) -> str:
        try:
            from scipy import stats
        except Exception as exc:
            raise RuntimeError(f"SciPy not available for KS tests: {exc}")

        numeric_cols = [
            c for c in df_full.columns
            if c in df_sample.columns and pd.api.types.is_numeric_dtype(df_full[c])
        ]
        rows = []
        for col in numeric_cols:
            x = df_full[col].replace([np.inf, -np.inf], np.nan).dropna()
            y = df_sample[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(x) < 10 or len(y) < 10:
                continue
            try:
                ks_stat, p_val = stats.ks_2samp(x, y)
            except Exception:
                ks_stat, p_val = 0.0, 0.0
            rows.append({"feature": col, "ks_stat": ks_stat, "p_value": p_val})

        ks_df = pd.DataFrame(rows).sort_values("p_value", ascending=True)
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"ks_validation_{name}.csv")
        ks_df.to_csv(csv_path, index=False)

        # Plot: KS statistic vs p-value
        fig, ax = plt.subplots(figsize=(10, 5))
        if ks_df.empty:
            ax.text(0.5, 0.5, "No KS data available", ha="center", va="center")
        else:
            ax.scatter(ks_df["ks_stat"], ks_df["p_value"], c="#3498db", alpha=0.7)
            ax.axhline(0.05, color="red", linestyle="--", label="p = 0.05")
            ax.set_xlabel("KS statistic")
            ax.set_ylabel("p-value")
            ax.set_title(f"KS test: validation vs sampled ({name})")
            ax.legend()
        plt.tight_layout()
        plot_path = os.path.join("graph", "stratification", f"ks_{name}.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Summary
        total = len(ks_df)
        pass_count = int((ks_df["p_value"] >= 0.05).sum()) if total else 0
        summary = (
            f"{name.upper()} KS summary: {pass_count}/{total} features with p >= 0.05"
            if total else f"{name.upper()} KS summary: no features tested"
        )
        return summary

    def run(self, context: DAGContext) -> TaskResult:
        start_ts = time.time()
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)

        try:
            cic_val = context.artifact_store.load_table("cic_validation")
            cic_sample = context.artifact_store.load_table("cic_consolidated")
            ton_val = context.artifact_store.load_table("ton_validation")
            ton_sample = context.artifact_store.load_table("ton_clean")
        except Exception as exc:
            return TaskResult(
                task_name=self.name,
                status="failed",
                duration_s=time.time() - start_ts,
                error=f"Missing validation artifacts: {exc}",
            )

        df_cic_val = context.table_io.read_parquet(cic_val.path).collect().to_pandas()
        df_cic_sample = context.table_io.read_parquet(cic_sample.path).collect().to_pandas()
        df_ton_val = context.table_io.read_parquet(ton_val.path).collect().to_pandas()
        df_ton_sample = context.table_io.read_parquet(ton_sample.path).collect().to_pandas()

        summaries = []
        summaries.append(self._ks_report(df_cic_val, df_cic_sample, "cic", report_dir))
        summaries.append(self._ks_report(df_ton_val, df_ton_sample, "ton", report_dir))

        report_path = os.path.join(report_dir, "sampling_ks_report.md")
        with open(report_path, "w") as f:
            f.write("# Sampling validation (KS test)\n\n")
            for line in summaries:
                f.write(f"- {line}\n")

        context.logger.info("sampling", "KS sampling validation complete", report=report_path)
        return TaskResult(
            task_name=self.name,
            status="ok",
            duration_s=time.time() - start_ts,
            outputs=[report_path],
        )
