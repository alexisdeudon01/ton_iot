from abc import ABC, abstractmethod
from typing import List
import polars as pl
from src.core.contracts.profiles import TableProfile, DistributionBundle

class ProfilingPort(ABC):
    @abstractmethod
    def profile_table(self, df: pl.LazyFrame, artifact_id: str, dataset: str, source_step: str) -> TableProfile:
        pass

    @abstractmethod
    def compute_distribution(self, df: pl.LazyFrame, artifact_id: str, feature: str) -> DistributionBundle:
        pass
