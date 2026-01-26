from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import pandas as pd

from alignment.selectors import FeatureSelector


EXCLUDE = {"y", "label", "Label", "type", "source_file", "sample_id", "ts"}


def extract_common_features(
    cic_df: pd.DataFrame,
    ton_df: pd.DataFrame,
    preferred: Optional[Iterable[str]] = None,
    limit: int = 15,
) -> List[str]:
    cic_cols = [c for c in cic_df.columns if c not in EXCLUDE]
    ton_cols = [c for c in ton_df.columns if c not in EXCLUDE]

    if preferred:
        preferred_list = [c for c in preferred if c in cic_cols and c in ton_cols]
        if len(preferred_list) >= limit:
            return preferred_list[:limit]

    common = sorted(set(cic_cols) & set(ton_cols))
    if len(common) >= limit:
        return common[:limit]

    # If not enough, keep whatever is available
    return common


def project_to_common(
    cic_df: pd.DataFrame,
    ton_df: pd.DataFrame,
    features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cic_proj = FeatureSelector.project_to_common(cic_df, features)
    ton_proj = FeatureSelector.project_to_common(ton_df, features)
    return cic_proj, ton_proj
