import polars as pl
from src.core.ports.table_io import TableIO

class PolarsIO(TableIO):
    def read_csv(self, path: str, **kwargs) -> pl.LazyFrame:
        return pl.scan_csv(path, **kwargs)

    def read_parquet(self, path: str) -> pl.LazyFrame:
        return pl.scan_parquet(path)

    def write_parquet(self, df: pl.DataFrame, path: str, compression: str = "zstd") -> None:
        df.write_parquet(path, compression=compression)

    def write_csv(self, df: pl.DataFrame, path: str) -> None:
        df.write_csv(path)

    def scan_csv(self, path: str, **kwargs) -> pl.LazyFrame:
        return pl.scan_csv(path, **kwargs)
