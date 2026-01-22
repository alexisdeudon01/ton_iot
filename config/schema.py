from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

class PathsConfig(BaseModel):
    cic_raw_dir: str
    ton_raw_path: str
    output_dir: str = "Outputs/output"
    data_dir: str = "data"

class MemoryConfig(BaseModel):
    max_ram_percent: float = 70.0
    safe_frac: float = 0.3
    row_group_size: int = 50000

class PreprocessConfig(BaseModel):
    impute_strategy: str = "median"
    scaling: str = "robust"
    use_cats: bool = False
    pca_variance: float = 0.95

class AlignmentConfig(BaseModel):
    cosine_threshold: float = 0.90
    ks_threshold: float = 0.05
    wasserstein_threshold: float = 10.0

class TrainingConfig(BaseModel):
    algorithms: List[str] = ["LR", "DT", "RF", "CNN", "TabNet"]
    test_size: float = 0.2
    val_size: float = 0.1
    seed: int = 42

class FusionConfig(BaseModel):
    method: Literal["weighted_avg"] = "weighted_avg"
    weights: Dict[str, float] = Field(default_factory=dict)
    threshold: float = 0.5

class XAIConfig(BaseModel):
    enabled: bool = True
    n_samples: int = 100
    top_k: int = 10

class PipelineConfig(BaseModel):
    paths: PathsConfig
    memory: MemoryConfig = MemoryConfig()
    preprocessing: PreprocessConfig = PreprocessConfig()
    alignment: AlignmentConfig = AlignmentConfig()
    training: TrainingConfig = TrainingConfig()
    fusion: FusionConfig = FusionConfig()
    xai: XAIConfig = XAIConfig()
    test_mode: bool = False
