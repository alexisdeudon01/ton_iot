from __future__ import annotations

from typing import List, Optional, Tuple

import polars as pl

from alignment.feature_alignment import FeatureAligner
from alignment.selectors import FeatureSelector


DEFAULT_EXCLUDE = {"y", "source_file", "sample_id", "Label", "type", "ts"}


def _fallback_common(cic_df: pl.DataFrame, ton_df: pl.DataFrame) -> List[str]:
    cic_cols = [c for c in cic_df.columns if c not in DEFAULT_EXCLUDE]
    ton_cols = [c for c in ton_df.columns if c not in DEFAULT_EXCLUDE]
    return sorted(list(set(cic_cols) & set(ton_cols)))


def align_features(
    cic_df: pl.DataFrame,
    ton_df: pl.DataFrame,
    alignment_cfg: Optional[dict] = None,
    fallback_features: Optional[List[str]] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame, List[str]]:
    common_features: List[str] = []
    try:
        from config.schema import AlignmentConfig

        cfg = AlignmentConfig(**(alignment_cfg or {}))
        aligner = FeatureAligner(cfg)
        common_features = aligner.compute_alignment_matrix(cic_df, ton_df)
    except Exception:
        common_features = []

    if not common_features:
        if fallback_features:
            common_features = [f for f in fallback_features if f in cic_df.columns and f in ton_df.columns]
        else:
            common_features = _fallback_common(cic_df, ton_df)

    cic_proj = FeatureSelector.project_to_common(cic_df, common_features)
    ton_proj = FeatureSelector.project_to_common(ton_df, common_features)
    return cic_proj, ton_proj, common_features
