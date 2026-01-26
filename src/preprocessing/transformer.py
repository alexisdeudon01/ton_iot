from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from preprocessing.preprocess_builder import PreprocessBuilder


def transform(df: pd.DataFrame, config: Dict | None = None) -> Tuple[pd.DataFrame, List[str]]:
    """Apply preprocessing (imputation, LogWinsorizer if enabled, RobustScaler, encoding).

    Uses existing PreprocessBuilder to preserve pipeline behavior.
    """
    builder = PreprocessBuilder(config or {})
    return builder.execute(df)
