import polars as pl
import numpy as np
from typing import Dict, Any
from src.core.ports.interfaces import IProfiler
from src.core.contracts.artifacts import TableProfile, DistributionBundle

class ProfilerAdapter(IProfiler):
    def profile_table(self, df: pl.LazyFrame, name: str, source_step: str) -> TableProfile:
        df_collected = df.collect()
        n_rows, n_cols = df_collected.shape
        columns = df_collected.columns
        dtypes = {col: str(dtype) for col, dtype in zip(columns, df_collected.dtypes)}
        
        missing_rate = {col: df_collected[col].null_count() / n_rows for col in columns}
        
        # Label balance (assuming 'y' is the label)
        label_balance = {}
        if "y" in df_collected.columns:
            counts = df_collected["y"].value_counts()
            label_balance = {str(row["y"]): row["count"] for row in counts.to_dicts()}

        numeric_summary = {}
        top_categories = {}
        
        for col in columns:
            series = df_collected[col]
            if series.dtype.is_numeric():
                desc = series.describe()
                # desc is a dataframe with 'statistic' and 'value'
                stats = {row["statistic"]: row["value"] for row in desc.to_dicts()}
                numeric_summary[col] = {
                    "min": stats.get("min", 0.0),
                    "max": stats.get("max", 0.0),
                    "mean": stats.get("mean", 0.0),
                    "std": stats.get("std", 0.0),
                    "median": series.median() or 0.0,
                    "q1": series.quantile(0.25) or 0.0,
                    "q3": series.quantile(0.75) or 0.0,
                }
            else:
                # Categorical
                counts = series.value_counts().sort("count", descending=True).head(10)
                top_categories[col] = {str(row[col]): row["count"] for row in counts.to_dicts()}

        return TableProfile(
            dataset="unknown", # Will be set by task
            name=name,
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

    def get_distribution(self, df: pl.LazyFrame, feature: str) -> DistributionBundle:
        series = df.select(feature).collect()[feature]
        
        if series.dtype.is_numeric():
            # Drop nulls for histogram
            data = series.drop_nulls().to_numpy()
            counts, bins = np.histogram(data, bins=20)
            
            quantiles = {
                "p1": series.quantile(0.01) or 0.0,
                "p5": series.quantile(0.05) or 0.0,
                "p50": series.quantile(0.50) or 0.0,
                "p95": series.quantile(0.95) or 0.0,
                "p99": series.quantile(0.99) or 0.0,
            }
            
            # Simple outlier detection (IQR)
            q1 = series.quantile(0.25) or 0.0
            q3 = series.quantile(0.75) or 0.0
            iqr = q3 - q1
            outliers = series.filter((pl.col(feature) < q1 - 1.5 * iqr) | (pl.col(feature) > q3 + 1.5 * iqr)).count()
            
            return DistributionBundle(
                artifact_id="unknown",
                feature=feature,
                bins=bins.tolist(),
                counts=counts.tolist(),
                quantiles=quantiles,
                outliers_count=outliers
            )
        else:
            # Categorical distribution
            counts_df = series.value_counts().sort("count", descending=True).head(20)
            return DistributionBundle(
                artifact_id="unknown",
                feature=feature,
                bins=[str(x) for x in counts_df[feature].to_list()],
                counts=counts_df["count"].to_list(),
                quantiles={},
                outliers_count=0
            )
