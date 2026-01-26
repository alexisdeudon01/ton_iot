from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import pandas as pd

from src.data.data_loader import DataLoader
from src.preprocessing.harmonizer import extract_common_features, project_to_common
from src.preprocessing.transformer import transform


def _stratified_sample(df: pd.DataFrame, label_col: str, frac: float, seed: int) -> pd.DataFrame:
    if label_col not in df.columns or frac >= 1.0 or len(df) == 0:
        return df
    frac = max(0.0, min(frac, 1.0))
    if frac == 0.0:
        return df.head(0)

    def _sample_group(group: pd.DataFrame) -> pd.DataFrame:
        n = max(1, int(math.ceil(len(group) * frac)))
        if n >= len(group):
            return group
        return group.sample(n=n, random_state=seed)

    return (
        df.groupby(label_col, group_keys=False)
        .apply(_sample_group)
        .reset_index(drop=True)
    )


def load_and_preprocess(config: Dict) -> Dict[str, Path]:
    data_cfg = config.get("data", {})
    auto_use_samples = bool(data_cfg.get("auto_use_samples", False))
    loader = DataLoader(config=config)
    cic_df, ton_df = loader.load_datasets(auto_use_samples=auto_use_samples)

    sampling_cfg = config.get("sampling", {})
    frac = sampling_cfg.get("fraction", 0.05)
    frac = float(frac)
    frac = min(frac, 0.05)
    seed = int(sampling_cfg.get("seed", 42))
    if frac < 1.0:
        cic_df = _stratified_sample(cic_df, "y", frac, seed)
        ton_df = _stratified_sample(ton_df, "y", frac, seed)

    features = extract_common_features(
        cic_df,
        ton_df,
        preferred=config.get("common_features"),
        limit=15,
    )

    cic_df, ton_df = project_to_common(cic_df, ton_df, features)

    cic_proc, feature_order = transform(cic_df, config.get("preprocessing"))
    ton_proc, _ = transform(ton_df, config.get("preprocessing"))

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    cic_path_out = processed_dir / "cic_processed.csv"
    ton_path_out = processed_dir / "ton_processed.csv"

    cic_proc.to_csv(cic_path_out, index=False)
    ton_proc.to_csv(ton_path_out, index=False)

    (processed_dir / "feature_order.json").write_text(
        json.dumps(feature_order, indent=2), encoding="utf-8"
    )
    (processed_dir / "common_features.json").write_text(
        json.dumps(features, indent=2), encoding="utf-8"
    )

    return {
        "cic_processed": cic_path_out,
        "ton_processed": ton_path_out,
        "feature_order": processed_dir / "feature_order.json",
        "common_features": processed_dir / "common_features.json",
    }


__all__ = ["load_and_preprocess"]
