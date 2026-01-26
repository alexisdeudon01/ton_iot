from __future__ import annotations

from typing import Dict, List, Tuple

import polars as pl

from preprocessing.preprocess_builder import PreprocessBuilder


def transform_dataset(df: pl.DataFrame, preprocessing_cfg: Dict | None = None) -> Tuple[pl.DataFrame, List[str]]:
    from config.schema import PreprocessConfig

    cfg = PreprocessConfig(**(preprocessing_cfg or {}))
    builder = PreprocessBuilder(cfg)
    return builder.execute(df)
