from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class PipelineConfig(BaseModel):
    # Paths
    cic_dir_path: str
    ton_csv_path: str
    work_dir: str
    artifacts_dir: str
    
    # IO
    parquet_compression: str = "zstd"
    row_group_size: int = 100_000
    
    # Sampling
    max_ram_percent: float = 80.0
    safe_frac_default: float = 0.1
    min_rows: int = 1000
    max_rows: Optional[int] = None
    sample_ratio: float = 1.0
    
    # Alignment
    descriptor_sample_rows: int = 5000
    cosine_min: float = 0.9
    ks_p_min: float = 0.05
    wasserstein_max: float = 0.1
    
    # Preprocessing
    use_cats: bool = True
    cat_strategy: Literal["onehot", "hash", "topk_other"] = "onehot"
    topk_k: int = 10
    
    # Clustering
    enable_clustering: bool = False
    reducer: Literal["pca", "umap", "none"] = "none"
    hdbscan_params: dict = Field(default_factory=lambda: {"min_cluster_size": 15})
    
    # Training
    algorithms: List[Literal["LR", "DT", "RF", "CNN", "TabNet"]] = ["LR", "DT", "RF", "CNN", "TabNet"]
    cv_folds: int = 5
    tuning_budget: int = 10
    
    # Fusion
    fusion_method: Literal["weighted_avg", "stacking_simple"] = "weighted_avg"
    weight_w: float = 0.5
    fusion_threshold: float = 0.5
    
    # Meta
    version: str = "1.0.0"
    seed: int = 42
    test_mode: bool = False
