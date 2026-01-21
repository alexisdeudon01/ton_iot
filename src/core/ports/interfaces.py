from typing import Protocol, List, Dict, Any, Optional, runtime_checkable
import polars as pl
from src.core.contracts.artifacts import TableArtifact, TableProfile, DistributionBundle, ModelArtifact
from src.core.events.models import PipelineEvent

@runtime_checkable
class ITableIO(Protocol):
    def read_table(self, path: str, columns: Optional[List[str]] = None) -> pl.LazyFrame:
        ...
    def write_table(self, df: pl.LazyFrame, path: str, format: str = "parquet") -> TableArtifact:
        ...

@runtime_checkable
class IModelAdapter(Protocol):
    def train(self, X: pl.DataFrame, y: pl.Series, params: Dict[str, Any]) -> Any:
        ...
    def predict_proba(self, model: Any, X: pl.DataFrame) -> pl.Series:
        ...
    def save(self, model: Any, path: str):
        ...
    def load(self, path: str) -> Any:
        ...

@runtime_checkable
class IProfiler(Protocol):
    def profile_table(self, df: pl.LazyFrame, name: str, source_step: str) -> TableProfile:
        ...
    def get_distribution(self, df: pl.LazyFrame, feature: str) -> DistributionBundle:
        ...

@runtime_checkable
class IArtifactStore(Protocol):
    def save_artifact(self, name: str, artifact: Any):
        ...
    def load_artifact(self, name: str) -> Any:
        ...
    def list_artifacts(self) -> List[str]:
        ...

@runtime_checkable
class IEventBus(Protocol):
    def publish(self, event: PipelineEvent):
        ...
    def subscribe(self, callback: Any):
        ...

@runtime_checkable
class ILogger(Protocol):
    def log(self, level: str, message: str, **kwargs):
        ...
