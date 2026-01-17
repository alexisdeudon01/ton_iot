import sys
import logging
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import unittest.mock as mock

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

pytest.importorskip("torch", reason="torch not available")

from src.config import PipelineConfig
from src.app.pipeline_runner import PipelineRunner

def setup_test_logging(output_dir: Path):
    """Setup logging to match the user's requested format."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"main_{timestamp}.log"

    # Reset logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("__main__")
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def test_cnn_pipeline_full_flow():
    """
    Full 5-phase pipeline integration test.
    Reproduces the high-level flow requested by the user.
    """
    test_results_dir = Path("output/test/full_pipeline_cnn")
    if test_results_dir.exists():
        import shutil
        shutil.rmtree(test_results_dir)
    test_results_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_test_logging(test_results_dir)

    # Initialize configuration
    config = PipelineConfig(
        test_mode=True,
        sample_ratio=0.0005,
        random_state=42,
        output_dir=str(test_results_dir),
        cic_max_files=3,
        phase1_search_enabled=True,
        phase2_enabled=True,
        phase3_enabled=True,
        phase4_enabled=True,
        phase5_enabled=True,
        phase3_algorithms=['Logistic_Regression', 'Decision_Tree', 'Random_Forest', 'CNN', 'TabNet'],
        phase3_cv_folds=2
    )

    # Mock generate_108_configs to return only 2 configs for speed
    with mock.patch('src.phases.phase1_config_search.generate_108_configs') as mock_gen:
        from src.config import generate_108_configs
        real_configs = generate_108_configs()
        mock_gen.return_value = real_configs[:2]

        # Initialize and run PipelineRunner
        runner = PipelineRunner(config)
        results = runner.run()

    # Verify all phases produced results
    assert 1 in results, "Phase 1 failed"
    assert 2 in results, "Phase 2 failed"
    assert 3 in results, "Phase 3 failed"
    assert 5 in results, "Phase 5 failed"

    # Check if all algorithms are in Phase 3 results
    eval_results = results[3]['evaluation_results']
    expected_algos = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'CNN', 'TabNet']
    for algo in expected_algos:
        assert algo in eval_results['model_name'].values, f"{algo} missing from evaluation"

    # Verify Phase 1 "best config" propagation
    assert runner.best_config is not None, "Best config not stored in runner"
    assert (test_results_dir / 'phase1_config_search' / 'best_config.json').exists()

if __name__ == "__main__":
    test_cnn_pipeline_full_flow()
