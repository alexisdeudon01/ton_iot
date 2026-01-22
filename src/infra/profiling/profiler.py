import polars as pl
import numpy as np
from typing import List, Dict, Tuple
from src.core.ports.profiling import ProfilingPort
from src.core.contracts.profiles import TableProfile, DistributionBundle

class PolarsProfiler(ProfilingPort):
    def profile_table(self, df: pl.LazyFrame, artifact_id: str, dataset: str, source_step: str) -> TableProfile:
        # Collect basic stats
        df_collected = df.collect()
        n_rows = df_collected.height
        n_cols = df_collected.width
        columns = df_collected.columns
        dtypes = {col: str(dtype) for col, dtype in zip(columns, df_collected.dtypes)}
        
        # Missing rate
        missing_rate = {col: df_collected[col].null_count() / n_rows for col in columns}
        
        # Label balance
        label_balance = {}
        if "y" in columns:
            counts = df_collected["y"].value_counts()
            for row in counts.to_dicts():
                label_balance[str(row["y"])] = row["count"]
        
        # Numeric summary
        numeric_summary = {}
        numeric_cols = [col for col, dtype in dtypes.items() if "Int" in dtype or "Float" in dtype]
        for col in numeric_cols:
            series = df_collected[col].drop_nulls()
            if series.len() > 0:
                numeric_summary[col] = {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "median": float(series.median()),
                    "q1": float(series.quantile(0.25)),
                    "q3": float(series.quantile(0.75))
                }
        
        # Top categories
        top_categories = {}
        cat_cols = [col for col, dtype in dtypes.items() if "String" in dtype or "Categorical" in dtype]
        for col in cat_cols:
            counts = df_collected[col].value_counts().sort("count", descending=True).head(10)
            top_categories[col] = [(str(row[col]), row["count"]) for row in counts.to_dicts()]

        return TableProfile(
            artifact_id=artifact_id,
            dataset=dataset,
            source_step=source_step,
            n_rows=n_rows,
            n_cols=n_cols,
            columns=columns,
            dtypes=dtypes,
            missing_rate=missing_rate,
            label_balance=label_balance,
            numeric_summary=numeric_summary,
            top_categories=top_categories
        )

    def compute_distribution(self, df: pl.LazyFrame, artifact_id: str, feature: str) -> DistributionBundle:
        series = df.select(feature).collect()[feature].drop_nulls()
        if series.len() == 0:
            return DistributionBundle(artifact_id=artifact_id, feature=feature, bins=[], counts=[], quantiles={}, outliers_count=0)
        
        counts, bins = np.histogram(series.to_numpy(), bins=20)
        quantiles = {
            "p1": float(series.quantile(0.01)),
            "p5": float(series.quantile(0.05)),
            "p50": float(series.quantile(0.50)),
            "p95": float(series.quantile(0.95)),
            "p99": float(series.quantile(0.99))
        }
        
        # Outliers (IQR method)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outliers = series.filter((pl.element() < q1 - 1.5 * iqr) | (pl.element() > q3 + 1.5 * iqr))
        
        return DistributionBundle(
            artifact_id=artifact_id,
            feature=feature,
            bins=bins.tolist(),
            counts=counts.tolist(),
            quantiles=quantiles,
            outliers_count=outliers.len()
        )
