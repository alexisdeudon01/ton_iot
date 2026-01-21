# Implementation Plan

[Overview]
Refactor the TON IoT ML Pipeline to ensure memory safety, robust error handling, and modular architecture following the Sixt rules and established workflow.

The current pipeline suffers from critical memory management issues, specifically uncontrolled Dask-to-Pandas conversions that lead to high RAM usage and potential Out-Of-Memory (OOM) errors. Additionally, the codebase lacks standardized error handling, consistent function return types, and proper separation of concerns between business logic and visualization. This implementation will introduce a `MemoryAwareProcessor` for safe data handling, a custom exception framework, standardized result objects, a centralized visualization service, and a validated configuration system using Pydantic. These changes will significantly improve the pipeline's reliability, maintainability, and performance.

[Types]  
Standardize data structures and return types using Python dataclasses and Pydantic models for configuration and results.

- `src/core/results.py`:
    - `TrainingResult`: Dataclass for model training outputs (success, time, history, error).
    - `ValidationResult`: Dataclass for hyperparameter tuning results (best params, scores).
    - `TestResult`: Dataclass for final evaluation metrics (accuracy, F1, precision, recall, AUC).
- `src/new_pipeline/config.py`:
    - `PipelineConfig`: Pydantic model for validated configuration (paths, algorithm settings, resource limits).

[Files]
Create new core utility files and refactor existing pipeline modules to integrate these utilities.

- New files:
    - `src/core/memory_manager.py`: Implements `MemoryAwareProcessor`.
    - `src/core/exceptions.py`: Defines custom pipeline exceptions.
    - `src/core/results.py`: Defines result dataclasses.
    - `src/evaluation/visualization_service.py`: Centralizes plotting logic.
- Modified files:
    - `src/new_pipeline/config.py`: Migrated to Pydantic.
    - `src/new_pipeline/trainer.py`: Refactored for memory safety, results, and visualization delegation.
    - `src/new_pipeline/validator.py`: Refactored for memory safety and results.
    - `src/new_pipeline/tester.py`: Refactored for memory safety and results.
    - `src/new_pipeline/xai_manager.py`: Refactored for memory safety and results.
    - `src/new_pipeline/main.py`: Updated to orchestrate the refactored components.

[Functions]
Introduce new functions for safe memory operations and refactor existing pipeline methods to return result objects and handle exceptions.

- `MemoryAwareProcessor.safe_compute(dask_df, operation)`:
    - Purpose: Safely convert Dask DataFrame to Pandas based on available RAM.
    - Parameters: `dask_df` (dd.DataFrame), `operation` (str).
    - Returns: `pd.DataFrame` (sampled or full).
- `PipelineTrainer.train_single(name, X_train, y_train) -> TrainingResult`:
    - Refactored to use `safe_compute`, handle `ModelTrainingError`, and return `TrainingResult`.
- `PipelineValidator.validate_tuning(...) -> ValidationResult`:
    - Refactored to use `safe_compute` and return `ValidationResult`.
- `PipelineTester.evaluate_all(...) -> List[TestResult]`:
    - Refactored to use `safe_compute` and return a list of `TestResult`.

[Changes]
Implement the refactoring in a dependency-ordered sequence, starting with core utilities and moving to pipeline modules.

1. **Core Foundations**:
    - Create `src/core/memory_manager.py` with `MemoryAwareProcessor`.
    - Create `src/core/exceptions.py` with custom exception hierarchy.
    - Create `src/core/results.py` with result dataclasses.
2. **Cross-Cutting Services**:
    - Create `src/evaluation/visualization_service.py` and migrate plotting logic from `trainer.py`, `validator.py`, etc.
    - Refactor `src/new_pipeline/config.py` to use Pydantic for validation.
3. **Module Refactoring**:
    - Update `src/new_pipeline/trainer.py` to integrate `MemoryAwareProcessor`, `VisualizationService`, and return `TrainingResult`.
    - Update `src/new_pipeline/validator.py` to integrate `MemoryAwareProcessor` and return `ValidationResult`.
    - Update `src/new_pipeline/tester.py` to integrate `MemoryAwareProcessor` and return `TestResult`.
    - Update `src/new_pipeline/xai_manager.py` to integrate `MemoryAwareProcessor`.
4. **Orchestration**:
    - Update `src/new_pipeline/main.py` to use the refactored classes and handle the new result objects.
5. **Verification**:
    - Run the pipeline in `--test-mode` to verify memory usage and overall functionality.

[Tests]
Verify the refactored pipeline using existing and new tests, focusing on memory safety and correct result propagation.

- Unit tests for `MemoryAwareProcessor` to ensure correct sampling logic.
- Integration test using `main.py --test-mode` to verify the full pipeline flow.
- Check RAM usage during execution (must stay < 70% in test mode).
- Verify all output files (plots, reports) are correctly generated in the `rr/` directory.
