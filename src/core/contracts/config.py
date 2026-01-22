from typing import Literal, List
from pydantic import BaseModel, Field, field_validator

class PathsConfig(BaseModel):
    cic_dir_path: str
    ton_csv_path: str
    work_dir: str
    artifacts_dir: str

class IOConfig(BaseModel):
    parquet_compression: str = "zstd"
    row_group_size: int = 100_000

class SamplingPolicy(BaseModel):
    max_ram_percent: float = 80.0
    safe_frac_default: float = 0.1
    min_rows: int = 1000
    max_rows: int = 1_000_000

class AlignmentConfig(BaseModel):
    descriptor_sample_rows: int = 5000
    cosine_min: float = 0.9
    ks_p_min: float = 0.05
    wasserstein_max: float = 0.1

class PreprocessingConfig(BaseModel):
    use_cats: bool = True
    cat_strategy: Literal["onehot", "hash", "topk_other"] = "topk_other"
    topk_k: int = 10

class ClusteringConfig(BaseModel):
    enable: bool = False
    reducer: Literal["pca", "umap", "none"] = "none"
    hdbscan_params: dict = Field(default_factory=dict)

class TrainingConfig(BaseModel):
    algorithms: List[str]
    cv_folds: int = 5
    tuning_budget: int = 10

    @field_validator("algorithms")
    @classmethod
    def validate_algorithms(cls, v: List[str]) -> List[str]:
        allowed = ["LR", "DT", "RF", "CNN", "TabNet"]
        if set(v) != set(allowed) or len(v) != len(allowed):
            raise ValueError(f"Algorithms must be exactly {allowed} (order independent in set, but all must be present)")
        # Ensure strict order as requested by user if needed, but set comparison is safer for presence.
        # User said: "refuser toute liste algorithms différente (set & order)".
        if v != allowed:
            raise ValueError(f"Algorithms must be exactly {allowed} in this specific order.")
        return v

class FusionConfig(BaseModel):
    method: Literal["weighted_avg", "stacking_simple"] = "weighted_avg"
    weight_w: float = 0.5
    threshold: float = 0.5

class PipelineConfig(BaseModel):
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
    test_mode: bool = False
    sample_ratio: float = 1.0
