import polars as pl
import os
import hashlib
from typing import List, Optional
from src.core.ports.interfaces import ITableIO
from src.core.contracts.artifacts import TableArtifact

class PolarsTableIOAdapter(ITableIO):
    def __init__(self, compression: str = "zstd", row_group_size: int = 100_000):
        self.compression = compression
        self.row_group_size = row_group_size

    def read_table(self, path: str, columns: Optional[List[str]] = None) -> pl.LazyFrame:
        if path.endswith(".parquet"):
            return pl.scan_parquet(path)
        elif path.endswith(".csv"):
            return pl.scan_csv(path, infer_schema_length=10000, ignore_errors=True)
        else:
            raise ValueError(f"Unsupported file format: {path}")

    def write_table(self, df: pl.LazyFrame, path: str, format: str = "parquet") -> TableArtifact:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Materialize to get stats and write
        df_collected = df.collect()
        
        if format == "parquet":
            df_collected.write_parquet(path, compression=self.compression, row_group_size=self.row_group_size)
        elif format == "csv":
            df_collected.write_csv(path)
        else:
            raise ValueError(f"Unsupported output format: {format}")

        n_rows, n_cols = df_collected.shape
        columns = df_collected.columns
        dtypes = {col: str(dtype) for col, dtype in zip(columns, df_collected.dtypes)}
        
        # Simple fingerprint based on path and size
        fingerprint = hashlib.md5(f"{path}_{n_rows}_{n_cols}".encode()).hexdigest()

        return TableArtifact(
            name=os.path.basename(path),
            path=path,
            format=format,
            n_rows=n_rows,
            n_cols=n_cols,
            columns=columns,
            dtypes=dtypes,
            version="1.0.0",
            source_step="unknown",
            fingerprint=fingerprint
        )
