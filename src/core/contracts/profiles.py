from typing import Literal, Dict, List, Tuple
from pydantic import BaseModel

class TableProfile(BaseModel):
    artifact_id: str
    dataset: Literal["cic", "ton"]
    source_step: str
    n_rows: int
    n_cols: int
    columns: List[str]
    dtypes: Dict[str, str]
    missing_rate: Dict[str, float]
    label_balance: Dict[str, int]  # keys "0", "1"
    numeric_summary: Dict[str, Dict[str, float]]  # col -> {min, max, mean, std, median, q1, q3}
    top_categories: Dict[str, List[Tuple[str, int]]]

class DistributionBundle(BaseModel):
    artifact_id: str
    feature: str
    bins: List[float]
    counts: List[int]
    quantiles: Dict[str, float]  # p1, p5, p50, p95, p99
    outliers_count: int
