import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

pytest.importorskip("pytorch_tabnet", reason="pytorch-tabnet not available")

from src.main_pipeline import IRPPipeline
from src.models.tabnet import TabNetClassifierWrapper
from src.evaluation_3d import Evaluation3D
from src.core.preprocessing_pipeline import StratifiedCrossValidator

def test_tabnet_pipeline_output():
    """
    Integration test for TabNet model through all pipeline phases.
    Verifies that outputs are generated in the output/test/tabnet directory.
    """
    # Define test output directory
    test_results_dir = Path("output/test/tabnet")

    # Initialize pipeline with test directory and small sample
    # sample_ratio=0.001 for very fast test
    pipeline = IRPPipeline(results_dir=str(test_results_dir), sample_ratio=0.001, random_state=42)

    # Phase 1: Preprocessing
    X, y, feature_names = pipeline.phase1_preprocessing()

    # Verify Phase 1 output
    assert (test_results_dir / 'phase1_preprocessing' / 'preprocessed_data.csv').exists()

    # Phase 3: Evaluation (Customized for TabNet only)
    evaluator = Evaluation3D(feature_names=feature_names)
    model = TabNetClassifierWrapper(max_epochs=2, batch_size=1024, seed=42, verbose=0)

    # Use 2-fold CV for speed in tests
    cv = StratifiedCrossValidator(n_splits=2, random_state=42)

    all_results = []
    splits = list(cv.split(X, y))
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        results = evaluator.evaluate_model(
            model, f"TabNet_fold{fold_idx+1}",
            X_train, y_train, X_test, y_test,
            compute_explainability=False # Faster for test
        )
        results['fold'] = fold_idx + 1
        all_results.append(results)

    # Average results
    avg_results = {
        'model_name': 'TabNet',
        'f1_score': np.mean([r['f1_score'] for r in all_results]),
        'accuracy': np.mean([r['accuracy'] for r in all_results]),
        'precision': np.mean([r['precision'] for r in all_results]),
        'recall': np.mean([r['recall'] for r in all_results]),
        'training_time_seconds': np.mean([r['training_time_seconds'] for r in all_results]),
        'memory_used_mb': np.mean([r['memory_used_mb'] for r in all_results]),
        'explainability_score': np.mean([r['explainability_score'] for r in all_results])
    }

    results_df = pd.DataFrame([avg_results])
    phase3_dir = test_results_dir / 'phase3_evaluation'
    phase3_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(phase3_dir / 'evaluation_results.csv', index=False)

    # Generate reports and visualizations
    evaluator.generate_algorithm_report('TabNet', avg_results, phase3_dir)
    evaluator.generate_dimension_visualizations(phase3_dir)

    # Verify Phase 3 outputs
    assert (phase3_dir / 'evaluation_results.csv').exists()
    assert (phase3_dir / 'algorithm_reports' / 'TabNet_report.md').exists()

    # Phase 5: Ranking
    ranking_results = pipeline.phase5_ranking(results_df)

    # Verify Phase 5 output
    assert (test_results_dir / 'phase5_ranking' / 'ranking_results.csv').exists()

if __name__ == "__main__":
    test_tabnet_pipeline_output()
