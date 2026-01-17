"""
Test that resource metrics are non-negative
"""
import sys
from pathlib import Path
import pytest
import numpy as np

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.evaluation.resources import measure_inference_latency


def test_latency_non_negative():
    """Test that latency is non-negative"""
    from sklearn.linear_model import LogisticRegression
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    X_dummy = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    y_dummy = np.array([0, 1])
    model.fit(X_dummy, y_dummy)
    
    latency = measure_inference_latency(model, X_dummy[:1], n_runs=5)
    assert latency >= 0, f"Latency should be non-negative (got {latency} ms)"
    assert isinstance(latency, float), f"Latency should be float (got {type(latency)})"
