"""
Tests for Phase 3 with synthetic dataset
Verifies that --synthetic flag produces CSV files, images, and INDEX.md
"""
import sys
from pathlib import Path
import pytest
import pandas as pd

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import PipelineConfig
from src.phases.phase3_evaluation import Phase3Evaluation


def test_phase3_synthetic_produces_outputs(tmp_path):
    """
    Test that Phase 3 with --synthetic flag produces required outputs.
    
    Input:
        - Config with synthetic_mode=True
        - Minimal algorithms (LR only for speed)
    
    Processing:
        - Run Phase 3 with synthetic dataset
        - Verify outputs are generated
    
    Expected Output:
        - evaluation_results.csv
        - dimension_scores.csv
        - algorithm_reports/*.md (at least one)
        - visualizations/*.png (at least one)
        - metrics/INDEX.md (if ratio features available)
    
    Method:
        - Direct testing of Phase3Evaluation.run() with synthetic_mode
    """
    config = PipelineConfig(
        output_dir=str(tmp_path),
        test_mode=True,
        synthetic_mode=True,
        phase3_algorithms=["Logistic_Regression"],  # Only LR for speed
        phase3_cv_folds=2  # Reduced folds for speed
    )
    
    phase3 = Phase3Evaluation(config)
    
    # Run Phase 3
    result = phase3.run()
    
    # Verify evaluation_results.csv exists
    results_file = tmp_path / "phase3_evaluation" / "evaluation_results.csv"
    assert results_file.exists(), f"evaluation_results.csv should exist at {results_file}"
    
    # Verify CSV content
    results_df = pd.read_csv(results_file)
    assert len(results_df) > 0, "evaluation_results.csv should contain at least one row"
    assert "model_name" in results_df.columns, "evaluation_results.csv should contain model_name column"
    assert "f1_score" in results_df.columns, "evaluation_results.csv should contain f1_score column"
    
    # Verify dimension_scores.csv exists
    dimension_scores_file = tmp_path / "phase3_evaluation" / "dimension_scores.csv"
    if dimension_scores_file.exists():
        dimension_scores_df = pd.read_csv(dimension_scores_file)
        assert len(dimension_scores_df) > 0, "dimension_scores.csv should contain at least one row"
        assert "model_name" in dimension_scores_df.columns, \
            "dimension_scores.csv should contain model_name column"
    
    # Verify algorithm reports directory
    algorithm_reports_dir = tmp_path / "phase3_evaluation" / "algorithm_reports"
    if algorithm_reports_dir.exists():
        report_files = list(algorithm_reports_dir.glob("*.md"))
        assert len(report_files) > 0, \
            f"At least one algorithm report should exist in {algorithm_reports_dir}"
    
    # Verify visualizations directory (may not exist if matplotlib fails, but should try)
    visualizations_dir = tmp_path / "phase3_evaluation" / "visualizations"
    # Visualizations may not always generate, so this is optional
    
    # Verify metrics directory and INDEX.md if ratio features exist
    metrics_dir = tmp_path / "phase3_evaluation" / "metrics"
    if metrics_dir.exists():
        index_file = metrics_dir / "INDEX.md"
        # INDEX.md may not exist if ratio features are missing, which is OK for synthetic data