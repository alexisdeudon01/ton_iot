import pytest
from pydantic import ValidationError
from src.core.contracts.config import PipelineConfig
from src.core.contracts.artifacts import TableArtifact

def test_pipeline_config_valid():
    config = PipelineConfig(
        cic_dir_path="data/cic",
        ton_csv_path="data/ton.csv",
        work_dir="out/work",
        artifacts_dir="out/art"
    )
    assert config.version == "1.0.0"
    assert "LR" in config.algorithms

def test_pipeline_config_invalid_algo():
    with pytest.raises(ValidationError):
        PipelineConfig(
            cic_dir_path="data/cic",
            ton_csv_path="data/ton.csv",
            work_dir="out/work",
            artifacts_dir="out/art",
            algorithms=["XGBoost"] # Forbidden
        )

def test_table_artifact_validation():
    art = TableArtifact(
        name="test",
        path="test.parquet",
        n_rows=100,
        n_cols=5,
        columns=["a", "b", "c", "d", "y"],
        dtypes={"a": "Int64", "y": "Int64"},
        version="1.0",
        source_step="step1",
        fingerprint="hash123"
    )
    assert art.n_rows == 100
