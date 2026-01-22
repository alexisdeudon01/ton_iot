from typing import Literal, List, Dict, Optional
from dataclasses import dataclass
from pydantic import BaseModel

@dataclass
class DatasetSpec:
    name: Literal["cic", "ton"]
    raw_paths: List[str]
    parquet_path: str
    label_rule: str
    id_cols_drop: List[str]

class TableArtifact(BaseModel):
    artifact_id: str
    name: str
    path: str
    format: Literal["parquet", "csv"]
    n_rows: int
    n_cols: int
    columns: List[str]
    dtypes: Dict[str, str]
    feature_order: Optional[List[str]] = None
    version: str
    source_step: str
    fingerprint: str
    stats: Dict

class AlignmentArtifact(BaseModel):
    artifact_id: str
    mapping_table: TableArtifact
    F_common: List[str]
    metrics_summary: Dict

class PreprocessArtifact(BaseModel):
    artifact_id: str
    preprocess_path: str
    num_features: List[str]
    cat_features: List[str]
    feature_order: List[str]
    steps: Dict
    version: str

class ModelArtifact(BaseModel):
    artifact_id: str
    model_path: str
    model_type: Literal["LR", "DT", "RF", "CNN", "TabNet"]
    dataset: Literal["cic", "ton"]
    feature_order: List[str]
    calibration: Literal["none", "platt", "isotonic"]
    metrics_cv: Dict
    version: str

class PredictionArtifact(BaseModel):
    artifact_id: str
    path: str
    format: Literal["parquet"] = "parquet"
    # required_columns: ["proba", "y_true", "dataset", "model", "split", "source_file"]
    version: str
