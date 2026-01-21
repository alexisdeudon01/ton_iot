from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from dataclasses import dataclass

@dataclass
class DatasetSpec:
    name: Literal["cic", "ton"]
    raw_paths: List[str]
    parquet_path: str
    label_rule: str
    id_cols_drop: List[str]

class TableArtifact(BaseModel):
    name: str
    path: str
    format: Literal["parquet", "csv"] = "parquet"
    n_rows: int
    n_cols: int
    columns: List[str]
    dtypes: Dict[str, str]
    feature_order: Optional[List[str]] = None
    version: str
    source_step: str
    fingerprint: str
    stats: Dict[str, Any] = Field(default_factory=dict)

class TableProfile(BaseModel):
    dataset: str
    name: str
    source_step: str
    n_rows: int
    n_cols: int
    columns: List[str]
    dtypes: Dict[str, str]
    missing_rate: Dict[str, float]
    label_balance: Dict[str, int]
    numeric_summary: Dict[str, Dict[str, float]] # col -> {min, max, mean, std, median, q1, q3}
    top_categories: Dict[str, Dict[str, int]] # col -> {val: count}

class DistributionBundle(BaseModel):
    artifact_id: str
    feature: str
    bins: List[float]
    counts: List[int]
    quantiles: Dict[str, float] # p1, p5, p50, p95, p99
    outliers_count: int

class AlignmentArtifact(BaseModel):
    mapping_table: TableArtifact
    F_common: List[str]
    metrics_summary: Dict[str, Any]

class PreprocessArtifact(BaseModel):
    preprocess_path: str
    num_features: List[str]
    cat_features: List[str]
    feature_order: List[str]
    steps_params: Dict[str, Any]
    version: str

class ModelArtifact(BaseModel):
    model_path: str
    model_type: Literal["LR", "DT", "RF", "CNN", "TabNet"]
    dataset: Literal["cic", "ton"]
    feature_order: List[str]
    calibration: Literal["none", "platt", "isotonic"] = "none"
    metrics_cv: Dict[str, Any]
    version: str

class PredictionArtifact(BaseModel):
    path: str
    columns: List[str] # ["proba", "y_true", "dataset", "model", "split", "source_file"]
    n_rows: int
    version: str
