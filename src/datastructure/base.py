from typing import Any
import pandas as pd
import dask.dataframe as dd

class IRPBaseStructure:
    """Base class for all custom data structures in the IRP project."""
    def __init__(self, *args, **kwargs):
        self.metadata = {}
        self.project_name = "IRP DDoS Detection"

    def add_metadata(self, key: str, value: Any):
        self.metadata[key] = value

class IRPDataFrame(pd.DataFrame, IRPBaseStructure):
    """Custom Pandas DataFrame for IRP."""
    def __init__(self, *args, **kwargs):
        pd.DataFrame.__init__(self, *args, **kwargs)
        IRPBaseStructure.__init__(self)

    @property
    def _constructor(self):
        return IRPDataFrame

class IRPDaskFrame(dd.DataFrame, IRPBaseStructure):
    """Custom Dask DataFrame for IRP."""
    def __init__(self, expr, name, meta, divisions):
        dd.DataFrame.__init__(self, expr, name, meta, divisions)
        IRPBaseStructure.__init__(self)
