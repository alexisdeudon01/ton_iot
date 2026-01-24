from typing import Literal, List, Optional
from pydantic import BaseModel, Field, field_validator

class PathsConfig(BaseModel):
    cic_dir_path: str
    ton_csv_path: str
    work_dir: str
    artifacts_dir: str

class IOConfig(BaseModel):
    parquet_compression: str = "zstd"
    row_group_size: int = 50000

class SamplingPolicy(BaseModel):
    max_ram_percent: float = 70.0
    safe_frac_default: float = 0.3
    min_rows: int = 1000
    max_rows: int = 500000

class AlignmentConfig(BaseModel):
    descriptor_sample_rows: int = 20000
    cosine_min: float = 0.95
    ks_p_min: float = 0.05
    wasserstein_max: float = 1e9

class PreprocessingConfig(BaseModel):
    use_cats: bool = False
    cat_strategy: Literal["onehot", "hash", "topk_other"] = "onehot"
    topk_k: int = 50

class ClusteringConfig(BaseModel):
    enable: bool = False
    reducer: Literal["pca", "umap", "none"] = "none"
    hdbscan_params: dict = Field(default_factory=dict)

class AlgorithmConfig(BaseModel):
    name: str
    key: str
    params: dict = Field(default_factory=dict)

class TrainingConfig(BaseModel):
    algorithms: List[str] = Field(default_factory=lambda: ["LR", "DT", "RF", "CNN", "TabNet"])
    cv_folds: int = 3
    tuning_budget: int = 5

    @field_validator("algorithms")
    @classmethod
    def validate_algorithms(cls, v: List[str]) -> List[str]:
        allowed = ["LR", "DT", "RF", "CNN", "TabNet"]
        # On vérifie que les algos demandés sont dans la liste autorisée
        for a in v:
            if a not in allowed:
                raise ValueError(f"Algorithm {a} is not supported. Allowed: {allowed}")
        return v

class FusionConfig(BaseModel):
    method: Literal["weighted_avg", "stacking_simple"] = "weighted_avg"
    weight_w: float = 0.5
    threshold: float = 0.5

class PipelineConfig(BaseModel):
    algorithms: List[AlgorithmConfig] = Field(default_factory=list)
    paths: PathsConfig
    io: IOConfig
    sampling_policy: SamplingPolicy
    alignment: AlignmentConfig
    preprocessing: PreprocessingConfig
    clustering: ClusteringConfig
    training: TrainingConfig
    fusion: FusionConfig
    version: str = "1.0.0"
    seed: int = 42
    test_mode: bool = True
    sample_ratio: float = 0.5
    test_row_limit_per_file: int = 2000
    test_max_files_per_dataset: int = 3
    test_max_rows_total_per_dataset: int = 50000
