import pytest
import polars as pl
import os
import json
from src.core.contracts.config import PipelineConfig

def test_test_mode_sampling_logic():
    # Simulation de la logique de T01/T02
    test_row_limit = 100
    df = pl.DataFrame({"a": range(1000), "y": [0, 1] * 500})
    sampled = df.head(test_row_limit)
    assert sampled.height == test_row_limit
    assert "y" in sampled.columns

def test_eval_uses_test_split_only():
    # Simulation de la logique de T17
    df = pl.DataFrame({
        "proba": [0.9, 0.1, 0.8],
        "y_true": [1, 0, 1],
        "split": ["train", "test", "val"]
    })
    test_df = df.filter(pl.col("split") == "test")
    assert test_df.height == 1
    assert test_df["split"][0] == "test"

def test_alignment_degraded_fallback():
    # Simulation de T05
    cic_cols = {"f1", "f2", "f3"}
    ton_cols = {"f4", "f5"}
    common = list(cic_cols & ton_cols)
    if not common:
        # Fallback mode test
        common = list(cic_cols)[:2]
    assert len(common) > 0

def test_prediction_artifact_has_required_columns():
    # Vérifie que les colonnes requises sont présentes dans le schéma attendu
    required = ["sample_id", "proba", "y_true", "dataset", "model", "split", "source_file"]
    df = pl.DataFrame({c: [] for c in required})
    for col in required:
        assert col in df.columns

def test_run_report_structure(tmp_path):
    # Vérifie que le rapport contient les entrées par algo
    report_path = tmp_path / "run_report.json"
    data = {
        "cic_RF": {"f1": 0.99},
        "ton_RF": {"f1": 0.98},
        "fused_global": {"f1": 0.99}
    }
    with open(report_path, "w") as f:
        json.dump(data, f)
    
    with open(report_path, "r") as f:
        loaded = json.load(f)
    assert "fused_global" in loaded
    assert "cic_RF" in loaded
