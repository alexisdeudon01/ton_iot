# Implementation Plan

[Overview]
Refactor the TON IoT ML Pipeline into a 3-layer architecture, ensuring memory safety, robust error handling, Parquet-based data processing, and joblib-based model persistence.

The current pipeline suffers from critical memory management issues and lacks a clear architectural separation. This implementation will restructure the codebase into three distinct layers: Business Logic (Pipeline), Data Structures, and Support (Monitoring/Utilities). It will also introduce a `MemoryAwareProcessor` for safe data handling, a custom exception framework, standardized result objects, and a centralized visualization service. Crucially, the data pipeline will be optimized by converting large CSV datasets to Parquet format for improved performance, and all trained models will be persisted using `joblib`.

[Types]  
Standardize data structures and return types using Python dataclasses and Pydantic models for configuration and results.

- **Layer 2: Data Structures**
    - `src/core/results.py`:
        - `TrainingResult`: Dataclass for model training outputs (success, time, history, error, model_path).
        - `ValidationResult`: Dataclass for hyperparameter tuning results (best params, scores).
        - `TestResult`: Dataclass for final evaluation metrics (accuracy, F1, precision, recall, AUC).
    - `src/datastructure/base.py` & `src/datastructure/flow.py`: Core data entities.
- **Layer 3: Support (Config)**
    - `src/new_pipeline/config.py`:
        - `PipelineConfig`: Pydantic model for validated configuration (paths, algorithm settings, resource limits).

[Files]
Create new core utility files and refactor existing pipeline modules to integrate these utilities and follow the 3-layer architecture.

- **Layer 1: Business Logic (Pipeline)**
    - `src/new_pipeline/data_loader.py`: Handles CSV to Parquet conversion and lazy loading.
    - `src/new_pipeline/trainer.py`: Refactored for memory safety, joblib persistence, and results.
    - `src/new_pipeline/validator.py`: Refactored for memory safety and results.
    - `src/new_pipeline/tester.py`: Refactored for memory safety and results.
    - `src/new_pipeline/xai_manager.py`: Refactored for memory safety and results.
    - `src/new_pipeline/main.py`: Orchestrates the 3-layer architecture.
- **Layer 2: Data Structures**
    - `src/core/results.py`: New file for result dataclasses.
    - `src/datastructure/`: Existing data structure files.
- **Layer 3: Support Layers**
    - `src/core/memory_manager.py`: New file for `MemoryAwareProcessor`.
    - `src/core/exceptions.py`: New file for custom pipeline exceptions.
    - `src/evaluation/visualization_service.py`: New file for centralized plotting.
    - `src/system_monitor.py`: Existing monitoring support.

[Functions]
Introduce new functions for safe memory operations, data conversion, and refactor existing pipeline methods.

- `RealDataLoader.convert_to_parquet()`:
    - Purpose: Convert large CSV datasets to Parquet format if not already present.
- `MemoryAwareProcessor.safe_compute(dask_df, operation)`:
    - Purpose: Safely convert Dask DataFrame to Pandas based on available RAM.
- `PipelineTrainer.train_single(name, X_train, y_train) -> TrainingResult`:
    - Refactored to use `safe_compute`, save models with `joblib`, and return `TrainingResult`.
- `PipelineValidator.validate_tuning(...) -> ValidationResult`:
    - Refactored to use `safe_compute` and return `ValidationResult`.

[Changes]
Implement the refactoring in a dependency-ordered sequence, following the 3-layer architecture.

1. **Support Layer (Layer 3) Foundations**:
    - Create `src/core/memory_manager.py`, `src/core/exceptions.py`.
    - Refactor `src/new_pipeline/config.py` to use Pydantic.
    - Create `src/evaluation/visualization_service.py`.
2. **Data Structure Layer (Layer 2)**:
    - Create `src/core/results.py` with result dataclasses.
    - Ensure `src/datastructure/` is properly integrated.
3. **Business Logic Layer (Layer 1) - Data Loading**:
    - Update `src/new_pipeline/data_loader.py` to implement CSV -> Parquet conversion and switch to `dd.read_parquet`.
4. **Business Logic Layer (Layer 1) - Pipeline Modules**:
    - Update `trainer.py` (joblib saving), `validator.py`, `tester.py`, `xai_manager.py`.
5. **Orchestration & Verification**:
    - Update `src/new_pipeline/main.py` and verify with `--test-mode`.

[Tests]
Verify the refactored pipeline focusing on Parquet integration, memory safety, and model persistence.

- Unit tests for `MemoryAwareProcessor` and `RealDataLoader` (Parquet conversion).
- Integration test using `main.py --test-mode`.
- Verify `.joblib` model files are created in the output directory.
- Check RAM usage during execution (must stay < 70% in test mode).

[Tests]
Verify the refactored pipeline using existing and new tests, focusing on memory safety and correct result propagation.

- Unit tests for `MemoryAwareProcessor` to ensure correct sampling logic.
- Integration test using `main.py --test-mode` to verify the full pipeline flow.
- Check RAM usage during execution (must stay < 70% in test mode).
- Verify all output files (plots, reports) are correctly generated in the `rr/` directory.
