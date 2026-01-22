from abc import ABC, abstractmethod
from typing import Any, List, Optional
import polars as pl

class TableIO(ABC):
    @abstractmethod
    def read_csv(self, path: str, **kwargs) -> pl.LazyFrame:
        pass

    @abstractmethod
    def read_parquet(self, path: str) -> pl.LazyFrame:
        pass

    @abstractmethod
    def write_parquet(self, df: pl.DataFrame, path: str, compression: str = "zstd") -> None:
        pass

    @abstractmethod
    def write_csv(self, df: pl.DataFrame, path: str) -> None:
        pass

    @abstractmethod
    def scan_csv(self, path: str, **kwargs) -> pl.LazyFrame:
        pass
