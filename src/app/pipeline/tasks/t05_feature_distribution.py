import os
import time
import joblib
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.app.pipeline.registry import TaskRegistry
from src.app.pipeline.universal_feature_mapping import OUTLIER_FEATURES, RATIO_FEATURES

from scipy.stats import ks_2samp
import pandas as pd

@TaskRegistry.register("T05_FeatureDistribution")
class T05_FeatureDistribution(Task):
    """
    Generates detailed distribution charts for TON, CIC, and comparisons.
    Also produces a feature analysis report.
    """
    def run(self, context: DAGContext) -> TaskResult:
        start_ts = time.time()
        base_dir = "./graph/feature_distributions"
        ton_dir = os.path.join(base_dir, "ton")
        cic_dir = os.path.join(base_dir, "cic")
        comp_dir = os.path.join(base_dir, "comparison")
        trans_dir = os.path.join(base_dir, "transformed_outliers")
        
        for d in [ton_dir, cic_dir, comp_dir, trans_dir]:
            os.makedirs(d, exist_ok=True)

        # 1. Load post-projection data (universal features)
        cic_art = context.artifact_store.load_table("cic_projected")
        ton_art = context.artifact_store.load_table("ton_projected")
        align_art = context.artifact_store.load_alignment("alignment_spec")

        df_cic = context.table_io.read_parquet(cic_art.path).collect()
        df_ton = context.table_io.read_parquet(ton_art.path).collect()

        exclude = {"y", "source_file", "sample_id"}
        common_features = [f for f in align_art.F_common if f in df_cic.columns and f in df_ton.columns]
        all_features = common_features

        # Prepare transformed versions (post-preprocessing)
        prep_cic = context.artifact_store.load_preprocess("preprocess_cic")
        prep_ton = context.artifact_store.load_preprocess("preprocess_ton")
        ct_cic = joblib.load(prep_cic.preprocess_path)
        ct_ton = joblib.load(prep_ton.preprocess_path)

        f_outlier = [f for f in OUTLIER_FEATURES if f in all_features]
        f_ratio = [f for f in RATIO_FEATURES if f in all_features]
        extra = [f for f in all_features if f not in f_outlier and f not in f_ratio]
        f_outlier.extend(extra)
        transformed_order = f_outlier + f_ratio
        index_map = {feat: idx for idx, feat in enumerate(transformed_order)}

        X_cic = df_cic.select(transformed_order).to_pandas()
        X_ton = df_ton.select(transformed_order).to_pandas()
        X_cic_t = ct_cic.transform(X_cic)
        X_ton_t = ct_ton.transform(X_ton)

        zero_variance = []
        different_dist = []

        # 2. Generation des graphiques
        for feat in all_features:
            has_cic = feat in df_cic.columns
            has_ton = feat in df_ton.columns
            
            data_cic = df_cic.select(feat).to_pandas()[feat].dropna() if has_cic else pd.Series()
            data_ton = df_ton.select(feat).to_pandas()[feat].dropna() if has_ton else pd.Series()
            
            is_numeric = pd.api.types.is_numeric_dtype(data_cic) or pd.api.types.is_numeric_dtype(data_ton)
            safe_feat = feat.replace("/", "_").replace(" ", "_").replace(":", "_")

            # --- Individual TON chart ---
            if has_ton and is_numeric and not data_ton.empty:
                plt.figure(figsize=(10, 6))
                sns.histplot(data_ton, kde=True, color="orange", stat="density")
                plt.title(f"Feature: {feat} (ToN-IoT)")
                plt.legend([f"Mean: {data_ton.mean():.2e}\nStd: {data_ton.std():.2e}\nn={len(data_ton)}"])
                plt.savefig(os.path.join(ton_dir, f"dist_{safe_feat}.png"))
                plt.close()
                if data_ton.std() == 0: zero_variance.append(f"TON: {feat}")

            # --- Individual CIC chart ---
            if has_cic and is_numeric and not data_cic.empty:
                plt.figure(figsize=(10, 6))
                sns.histplot(data_cic, kde=True, color="blue", stat="density")
                plt.title(f"Feature: {feat} (CIC-DDoS2019)")
                plt.legend([f"Mean: {data_cic.mean():.2e}\nStd: {data_cic.std():.2e}\nn={len(data_cic)}"])
                plt.savefig(os.path.join(cic_dir, f"dist_{safe_feat}.png"))
                plt.close()
                if data_cic.std() == 0: zero_variance.append(f"CIC: {feat}")

            # --- Comparative chart (bonus) ---
            if has_cic and has_ton and is_numeric and not data_cic.empty and not data_ton.empty:
                plt.figure(figsize=(12, 7))
                sns.kdeplot(data_cic, label=f"CIC (mean={data_cic.mean():.2e})", color="blue", fill=True, alpha=0.3)
                sns.kdeplot(data_ton, label=f"TON (mean={data_ton.mean():.2e})", color="orange", fill=True, alpha=0.3)
                plt.title(f"Comparison: {feat} (Domain Shift Visualization)")
                plt.legend()
                plt.savefig(os.path.join(comp_dir, f"compare_{safe_feat}.png"))
                plt.close()
                
                # Kolmogorov-Smirnov test to detect distribution differences
                stat, p = ks_2samp(data_cic, data_ton)
                if p < 0.01: different_dist.append(feat)

            # --- Before/After chart (outliers) ---
            if feat in f_outlier and feat in index_map:
                idx = index_map[feat]
                trans_cic = pd.Series(X_cic_t[:, idx])
                trans_ton = pd.Series(X_ton_t[:, idx])
                if not trans_cic.empty and not trans_ton.empty:
                    plt.figure(figsize=(12, 8))
                    plt.subplot(2, 1, 1)
                    sns.histplot(data_cic, kde=True, color="blue", stat="density", alpha=0.5, label="CIC raw")
                    sns.histplot(data_ton, kde=True, color="orange", stat="density", alpha=0.5, label="TON raw")
                    plt.title(f"{feat} - Before (raw)")
                    plt.legend()

                    plt.subplot(2, 1, 2)
                    sns.histplot(trans_cic, kde=True, color="blue", stat="density", alpha=0.5, label="CIC transformed")
                    sns.histplot(trans_ton, kde=True, color="orange", stat="density", alpha=0.5, label="TON transformed")
                    plt.title(f"{feat} - After (LogWinsorizer + RobustScaler)")
                    plt.legend()

                    plt.tight_layout()
                    plt.savefig(os.path.join(trans_dir, f"before_after_{safe_feat}.png"))
                    plt.close()

        # 3. Generate report
        report_path = os.path.join(base_dir, "feature_analysis_report.md")
        with open(report_path, "w") as f:
            f.write("# Feature Analysis Report (CIC vs TON)\n\n")
            f.write(f"## 1. Common Features ({len(common_features)})\n")
            f.write(", ".join(common_features) + "\n\n")
            f.write("## 2. Zero-variance Features (std=0)\n")
            f.write("- " + "\n- ".join(zero_variance) if zero_variance else "None")
            f.write("\n\n## 3. Features with Significantly Different Distributions (KS Test p < 0.01)\n")
            f.write("- " + "\n- ".join(different_dist) if different_dist else "None")

        context.logger.info("visualization", f"Analysis complete. Reports in {base_dir}")
        return TaskResult(task_name=self.name, status="ok", duration_s=time.time() - start_ts, outputs=[report_path])
