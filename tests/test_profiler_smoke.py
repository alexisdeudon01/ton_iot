import polars as pl
from src.infra.profiling.profiler import PolarsProfiler

def test_profiler_smoke():
    df = pl.DataFrame({
        "feat1": [1.0, 2.0, None, 4.0],
        "feat2": ["A", "B", "A", "C"],
        "y": [0, 1, 0, 1]
    }).lazy()
    
    profiler = PolarsProfiler()
    profile = profiler.profile_table(df, "test_art", "cic", "step1")
    
    assert profile.n_rows == 4
    assert profile.n_cols == 3
    assert "feat1" in profile.numeric_summary
    assert "feat2" in profile.top_categories
    assert profile.label_balance == {"0": 2, "1": 2}
    assert profile.missing_rate["feat1"] == 0.25
