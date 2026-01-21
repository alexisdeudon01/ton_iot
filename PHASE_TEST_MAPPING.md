# Phase -> Step -> Test Mapping (New Pipeline V7)

This document maps each phase and step of the New DDoS Detection Pipeline to its corresponding tests.

## Phase 1: Data Loading & Preprocessing
- **Step 1.1: Multi-dataset Loading (ToN-IoT & CICDDoS2019)**: Covered by `tests/test_new_pipeline_components.py` (Data Loader).
- **Step 1.2: Filtering & Mapping (Normal/DDoS only)**: Covered by `tests/test_new_pipeline_components.py`.
- **Step 1.3: Feature Categorization**: Covered by `tests/test_new_pipeline_components.py` (Categorization).
- **Step 1.4: Resource-Aware Loading (50% RAM)**: Covered by `tests/test_performance_and_ram.py`.

## Phase 2: Iterative Training
- **Step 2.1: Single Model Training (KNN, LR, DT, RF, CNN, TabNet)**: Covered by `tests/test_algo_handling.py` and `tests/test_new_pipeline_components.py`.
- **Step 2.2: Convergence Tracking**: Covered by `tests/test_new_pipeline_components.py`.

## Phase 3: Validation & Tuning
- **Step 3.1: Dynamic Hyperparameter Tuning**: Covered by `tests/test_new_pipeline_components.py` (Validator).
- **Step 3.2: Performance Metrics (F1, AUC)**: Covered by `tests/test_evaluation_3d_comprehensive.py`.

## Phase 4: XAI Validation
- **Step 4.1: Fidelity, Stability, Complexity Scores**: Covered by `tests/test_new_pipeline_components.py` (XAIManager).
- **Step 4.2: SHAP/LIME Visualizations**: Covered by `tests/_legacy_tests/test_explainability.py`.

## Phase 5: Final Testing
- **Step 5.1: Consolidated Results Report**: Covered by `tests/test_new_pipeline_components.py` (Tester).
- **Step 5.2: Resource Consumption Plots**: Covered by `tests/test_performance_and_ram.py`.

## Cross-Cutting Concerns
- **Data Leakage**: Covered by `tests/test_no_data_leakage.py` and `tests/test_test_transform_is_fold_fitted.py`.
- **RAM Optimization**: Covered by `tests/_legacy_tests/test_dataset_loader_oom_fix.py`.
- **Algorithm Handling**: Covered by `tests/test_algo_handling.py`.
