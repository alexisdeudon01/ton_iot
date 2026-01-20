# Phase -> Step -> Test Mapping

This document maps each phase and step of the IRP Research Pipeline to its corresponding tests to ensure full coverage.

## Phase 1: Preprocessing Configuration Search
- **Step 1.1: Loading and Harmonizing**: Covered by `tests/test_preprocessing_pipeline.py` (basic flow) and `tests/_legacy_tests/test_dataset_loader_oom_fix.py`.
- **Step 1.2: Early Fusion**: Covered by `tests/test_dataset_source_flag.py`.
- **Step 1.3: Evaluating 108 Configs**: Covered by `tests/_legacy_tests/test_phase1_108_configs.py` and `tests/_legacy_tests/test_phase1_config_search.py`.

## Phase 2: Apply Best Configuration
- **Step 2.1: Load and Harmonize**: Covered by `tests/test_phase2_outputs.py`.
- **Step 2.2: Stateless Preprocessing**: Covered by `tests/test_preprocessing_pipeline.py` (stateless checks).
- **Step 2.3: Save Outputs**: Covered by `tests/test_phase2_outputs.py`.

## Phase 3: Comprehensive Evaluation
- **Step 3.1: Load and Prepare**: Covered by `tests/test_phase3_synthetic.py`.
- **Step 3.2: Model-Aware Preprocessing per Fold**: Covered by `tests/test_model_aware_profiles.py` and `tests/test_no_smote_leakage_phase3.py`.
- **Step 3.3: Model Training and Evaluation**: Covered by `tests/test_cnn.py`, `tests/test_tabnet.py`, and `tests/test_phase3_cnn_tabnet.py`.
- **Step 3.4: Ratio Validation and KDE**: Covered by `tests/test_evaluation_3d_comprehensive.py`.

## Phase 4: AHP Preferences
- **Step 4.1: Define Preferences**: Covered by `tests/test_ahp_topsis.py`.

## Phase 5: TOPSIS Ranking
- **Step 5.1: Rank Models**: Covered by `tests/test_ahp_topsis.py`.

## Cross-Cutting Concerns
- **Data Leakage**: Covered by `tests/test_no_data_leakage.py` and `tests/test_test_transform_is_fold_fitted.py`.
- **RAM Optimization**: Covered by `tests/_legacy_tests/test_dataset_loader_oom_fix.py`.
- **Algorithm Handling**: Covered by `tests/test_algo_handling.py`.
