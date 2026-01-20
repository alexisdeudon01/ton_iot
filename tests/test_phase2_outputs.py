"""
Tests for Phase 2 outputs
Verifies that Phase 2 generates all required output files (parquet/csv.gz, feature_names.json, phase2_summary.md)
and that dataset_source is present and encoded (0/1)
"""
import sys
from pathlib import Path
import pytest
import pandas as pd
import json

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import PipelineConfig
from src.phases.phase2_apply_best_config import Phase2ApplyBestConfig


def test_phase2_outputs(tmp_path, monkeypatch):
    """
    Test that Phase 2 generates all required output files.
    
    Input:
        - Mock config and best_config
        - Mock dataset with dataset_source column
    
    Processing:
        - Run Phase 2 with mocked dataset loading
        - Verify output files are created
    
    Expected Output:
        - best_preprocessed.parquet (or csv.gz)
        - feature_names.json
        - phase2_summary.md
        - dataset_source column present and encoded (0/1)
    
    Method:
        - Direct testing of Phase2ApplyBestConfig.run()
    """
    config = PipelineConfig(output_dir=str(tmp_path), test_mode=True, sample_ratio=0.01)
    phase2 = Phase2ApplyBestConfig(config, best_config={"dummy": True})

    def fake_load_and_harmonize():
        data = {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10.0, 20.0, 30.0, 40.0],
            "dataset_source": [0, 0, 1, 1],  # CIC=0, TON=1
            "label": [0, 1, 0, 1],
        }
        return pd.DataFrame(data)

    monkeypatch.setattr(phase2, "_load_and_harmonize_datasets", fake_load_and_harmonize)

    result = phase2.run()

    # Verify output paths
    output_paths = result["output_paths"]
    assert output_paths["preprocessed_data"].exists(), \
        f"Preprocessed data file should exist at {output_paths['preprocessed_data']}"
    assert output_paths["feature_names"].exists(), \
        f"Feature names file should exist at {output_paths['feature_names']}"
    assert output_paths["summary"].exists(), \
        f"Summary file should exist at {output_paths['summary']}"

    # Verify preprocessed data contains dataset_source encoded as 0/1
    preprocessed_file = output_paths["preprocessed_data"]
    if preprocessed_file.suffix == ".parquet":
        df = pd.read_parquet(preprocessed_file)
    else:
        df = pd.read_csv(preprocessed_file, compression="gzip" if preprocessed_file.suffix == ".gz" else None)

    assert "dataset_source" in df.columns, "dataset_source column should be present in preprocessed data"
    assert df["dataset_source"].isin([0, 1]).all(), \
        f"dataset_source should be encoded as 0/1, got values: {df['dataset_source'].unique()}"
    assert "label" in df.columns, "label column should be present in preprocessed data"

    # Verify feature_names.json
    with open(output_paths["feature_names"]) as f:
        feature_data = json.load(f)
    assert "feature_names" in feature_data, "feature_names.json should contain 'feature_names' key"
    assert "dataset_source" in feature_data["feature_names"] or "dataset_source" not in df.drop(columns=["label"]).columns, \
        "dataset_source should be included in feature names if present"

    # Verify summary.md contains expected information
    summary_content = output_paths["summary"].read_text()
    assert "Phase 2: Apply Best Configuration" in summary_content, \
        "Summary should contain Phase 2 title"
    assert "dataset_source" in summary_content or "Dataset Source Distribution" in summary_content, \
        "Summary should document dataset_source mapping"
    assert "Stateless preprocessing only" in summary_content, \
        "Summary should mention stateless preprocessing"