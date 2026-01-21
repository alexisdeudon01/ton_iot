import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict, field
import io
import traceback
import inspect
import ast
import random

try:
    import pytest
except ImportError:
    print("❌ pytest not installed. Install with: pip install pytest")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib/pandas not available - diagram generation will be skipped")

class SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that avoids writing to closed streams."""
    def emit(self, record):
        try:
            if self.stream and not self.stream.closed:
                msg = self.format(record)
                self.stream.write(msg + self.terminator)
                self.flush()
        except (ValueError, AttributeError, OSError):
            pass

def setup_detailed_logging():
    """Configure logging with a safe handler."""
    root = logging.getLogger()
    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = SafeStreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(logging.INFO)

setup_detailed_logging()
logger = logging.getLogger(__name__)

@dataclass
class DatasetInfo:
    """Detailed information about a dataset"""
    name: str
    headers: List[str]
    header_row: Dict[str, Any]  # First row values
    random_row: Dict[str, Any]  # Random row (between row 2 and end)
    random_row_index: int  # Index of the random row
    shape: Tuple[int, int]
    dtype: str = ""
    raw_sample: Optional[pd.DataFrame] = None

@dataclass
class FusionInfo:
    """Information about dataset fusion process"""
    source_datasets: List[str]
    fusion_method: str
    fused_headers: List[str]
    fused_sample_row: Dict[str, Any]
    validation_method: str
    validation_results: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MatrixInfo:
    """Information about a matrix/DataFrame input"""
    name: str
    headers: List[str]
    sample_row: Dict[str, Any]
    shape: Tuple[int, int]
    dtype: str = ""
    datasets: List[DatasetInfo] = field(default_factory=list)
    fusion: Optional[FusionInfo] = None

@dataclass
class ValidationCriterion:
    """A validation criterion (assertion) from test code"""
    description: str
    condition: str  # The actual assertion condition

@dataclass
class TestResult:
    """Detailed test result with input/output tracking"""
    test_name: str
    outcome: str  # 'passed', 'failed', 'skipped', 'error'
    duration: float = 0.0
    input_description: str = ""
    output_description: str = ""
    input_matrices: List[MatrixInfo] = field(default_factory=list)
    validation_criteria: List[ValidationCriterion] = field(default_factory=list)
    success_reason: str = ""
    failure_reason: str = ""
    error_message: str = ""
    traceback: str = ""

class DetailedTestPlugin:
    """Pytest plugin to capture detailed test information"""

    def __init__(self):
        self.results: List[TestResult] = []
        self.test_start_times: Dict[str, float] = {}
        self.test_source_code: Dict[str, str] = {}
        self.test_fixtures: Dict[str, Dict[str, Any]] = {}

    def pytest_collection_modifyitems(self, config, items):
        for item in items:
            try:
                source = inspect.getsource(item.function)
                self.test_source_code[item.nodeid] = source
            except Exception:
                pass

    def pytest_runtest_setup(self, item):
        import time
        self.test_start_times[item.nodeid] = time.time()
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 SETUP: {item.nodeid}")
        logger.info(f"{'='*80}")

    def pytest_runtest_logreport(self, report):
        import time
        if report.when == "call":
            test_name = report.nodeid
            start_time = self.test_start_times.get(test_name, time.time())
            duration = time.time() - start_time
            outcome = report.outcome

            result = TestResult(
                test_name=test_name,
                outcome=outcome,
                duration=duration
            )
            self.results.append(result)

def main() -> int:
    test_mode = "-test" in sys.argv
    logger.info("=" * 80)
    logger.info("🧪 IRP PIPELINE TEST SUITE")
    logger.info("🚀 MODE: " + ("TEST" if test_mode else "FULL"))
    logger.info("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output/test_reports") / f"test_run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    plugin = DetailedTestPlugin()
    # Run only new tests to avoid legacy failures if requested, or all
    exit_code = pytest.main(["-v", "tests/test_new_pipeline_components.py"], plugins=[plugin])

    # Re-setup logging after pytest might have messed with it
    setup_detailed_logging()

    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST EXECUTION SUMMARY")
    logger.info("=" * 80)
    for result in plugin.results:
        logger.info(f"{'✅' if result.outcome == 'passed' else '❌'} {result.test_name} ({result.duration:.3f}s)")

    return exit_code

if __name__ == "__main__":
    sys.exit(main())
