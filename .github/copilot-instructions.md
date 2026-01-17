# TON IoT IRP Research Pipeline - AI Coding Agent Instructions

## Project Overview

This is an **IoT Intrusion Detection System (IDS)** research pipeline implementing the **IRP (Integrated Research Pipeline)** methodology for evaluating ML/DL algorithms across 3 dimensions: Detection Performance, Resource Efficiency, and Explainability. Published in 2024 IEEE ORSS.

**Core Workflow**: Multi-dataset fusion (CIC-DDoS2019 + TON_IoT) → 108 preprocessing configs search → Multi-dimensional evaluation → AHP-TOPSIS ranking.

## Architecture

### Entry Point & Module Structure

- **Single entry point**: [`main.py`](main.py) (project root) - adds `src/` to path, handles CLI args
- **All modules live in [`src/`](src/)** - never import from project root except `main.py`
- **Phases**: [`src/phases/phase{1,2,3,4,5}_*.py`](src/phases/) - one file per pipeline phase
- **Core modules**: [`src/core/`](src/core/) - dataset loading, harmonization, preprocessing, feature engineering
- **Models**: [`src/models/`](src/models/) - `registry.py` (single source of truth), `sklearn_models.py`, `cnn.py`, `tabnet.py`
- **Evaluation**: [`src/evaluation_3d.py`](src/evaluation_3d.py) - 3D evaluation framework with metrics/visualizations

### 5-Phase Pipeline (Sequential)

1. **Phase 1: Config Search** - Generate and evaluate 108 preprocessing configs (cleaning, encoding, feature selection, scaling, SMOTE combinations)
2. **Phase 2: Apply Best Config** - Apply winning config with **stateless preprocessing only** (cleaning + encoding), no fit-dependent steps to prevent data leakage
3. **Phase 3: Model-Aware Evaluation** - 3D evaluation with **model-aware preprocessing profiles** (LR: scaling+FS+SMOTE, Trees: class_weight='balanced', NN: scaling+SMOTE)
4. **Phase 4: AHP Preferences** - Define dimension weights (interactive or defaults)
5. **Phase 5: TOPSIS Ranking** - Multi-criteria decision making for final algorithm ranking

Run with: `python main.py [--phase N] [--test-mode] [--sample-ratio 0.001] [--synthetic]`

### Critical Anti-Patterns

- **Data Leakage Prevention**: Phase 2 performs ONLY stateless preprocessing (cleaning, encoding). Fit-dependent operations (scaling, feature selection, SMOTE) happen in Phase 3 **per CV fold** using `PreprocessingPipeline.transform_test()` methodology
- **Model-Aware Profiles**: Different models need different preprocessing in Phase 3 - see [`src/config.py::preprocessing_profiles`](src/config.py)
- **Optional Dependencies**: CNN/TabNet/SHAP/LIME are optional via `requirements-nn.txt`. Always check `*_AVAILABLE` flags before using

## Key Patterns

### Optional Dependency Pattern

```python
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    # Implementation here
else:
    logger.warning("Feature skipped (torch not available)")
```

Used for: torch, pytorch-tabnet, shap, lime. Core pipeline works with `requirements-core.txt` alone.

### Model Registry Pattern

[`src/models/registry.py`](src/models/registry.py) is the **single source of truth** for all models. Always use `get_model_registry(config)` to retrieve models. Registry automatically handles optional models (CNN, TabNet) and logs warnings if dependencies missing.

### Dataset Loading Strategy

- **Memory safety**: Uses chunked loading with automatic dtype optimization ([`src/core/dataset_loader.py`](src/core/dataset_loader.py))
- **Test mode**: `--test-mode` or `--sample-ratio 0.001` samples data, limits CIC files to 3
- **Early fusion**: Adds `dataset_source` feature (0=CIC-DDoS2019, 1=TON_IoT) encoded as integer during harmonization
- **Feature engineering**: Behavioral ratios (`Flow_Bytes_s`, `Flow_Packets_s`, `Avg_Packet_Size`, `Traffic_Direction_Ratio`) computed **before** harmonization

### Configuration System

- [`src/config.py`](src/config.py) centralizes all constants, thresholds, weights
- `PipelineConfig` dataclass for production, `TEST_CONFIG` for tests
- Never hardcode parameters - always reference config

### Testing Strategy

- **Fixtures**: [`tests/conftest.py`](tests/conftest.py) provides `config` and `test_config` fixtures
- **Test categories**:
  - Data leakage prevention (`test_no_data_leakage.py`)
  - Optional dependency handling (`test_imports_no_gui_dependency.py`, `test_registry.py`)
  - Model-aware profiles (`test_model_aware_profiles.py`)
  - Memory safety (`test_dataset_loader_oom_fix.py`)
  - Phase outputs (`test_phase1_108_configs.py`, `test_phase2_outputs.py`)
- Run: `python -m pytest tests/ -v`

## Project-Specific Commands

```bash
# Install (modular)
pip install -r requirements-core.txt          # Minimal (sklearn, pandas, imbalanced-learn)
pip install -r requirements-nn.txt            # Optional (torch, tabnet, shap, lime)

# Run full pipeline
python main.py

# Run specific phase
python main.py --phase 3

# Test mode (0.1% data, 3 CIC files max)
python main.py --test-mode

# Custom sampling
python main.py --sample-ratio 0.001 --cic-max-files 5

# Synthetic data (Phase 3 only, for testing)
python main.py --phase 3 --synthetic

# Tests
python -m pytest tests/ -v
python -m pytest tests/test_no_data_leakage.py -v
```

## Output Structure

All outputs in [`output/`](output/):

- `phase1_config_search/` - Best config JSON, comparison CSV
- `phase1_preprocessing/` - Preprocessed data (best config applied)
- `phase3_evaluation/` - Evaluation results, 27 PNG visualizations, algorithm reports (MD)
- `phase5_ranking/` - Final TOPSIS ranking CSV/MD
- `logs/` - Phase-specific logs with timestamps

## Common Tasks

### Adding a New Model

1. Create model in [`src/models/`](src/models/) following sklearn API (fit/predict/predict_proba)
2. Add factory function to [`src/models/registry.py`](src/models/registry.py)
3. Add model name to [`src/config.py::phase3_algorithms`](src/config.py)
4. Define preprocessing profile in [`src/config.py::preprocessing_profiles`](src/config.py) if needed

### Adding a New Preprocessing Step

1. Update [`src/core/preprocessing_pipeline.py::PreprocessingPipeline`](src/core/preprocessing_pipeline.py)
2. Add step to Phase 1 config generation in [`src/phases/phase1_config_search.py`](src/phases/phase1_config_search.py)
3. Update [`src/config.py::preprocessing_options`](src/config.py) for 108 config search
4. Add tests in [`tests/test_phase1_*.py`](tests/)

### Adding a New Dimension Metric

1. Update [`src/evaluation_3d.py`](src/evaluation_3d.py) - add metric calculation in respective dimension method
2. Update [`DIMENSIONS_CALCULATION.md`](DIMENSIONS_CALCULATION.md) with formula documentation
3. Add visualization in [`src/evaluation_3d.py::generate_dimension_visualizations()`](src/evaluation_3d.py)
4. Update [`IRP_COMPLIANCE_REVIEW.md`](IRP_COMPLIANCE_REVIEW.md) compliance checks

## Critical Files for Understanding

- [`README.md`](README.md) - Comprehensive methodology, installation, usage
- [`IRP_COMPLIANCE_REVIEW.md`](IRP_COMPLIANCE_REVIEW.md) - Maps implementation to research paper
- [`DIMENSIONS_CALCULATION.md`](DIMENSIONS_CALCULATION.md) - Detailed dimension formulas
- [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) - French file structure documentation
- [`src/config.py`](src/config.py) - All tuneable parameters
- [`src/models/registry.py`](src/models/registry.py) - Model availability logic

## Conventions

- **Logging**: Use module-level `logger = logging.getLogger(__name__)`, never print()
- **Paths**: Always use `pathlib.Path`, never string concatenation
- **Randomness**: Always use `config.random_state` for reproducibility
- **CSV Output**: Use `compression='gzip'` for large files, `.parquet` preferred for preprocessed data
- **Warnings**: Suppress sklearn warnings with `warnings.filterwarnings('ignore')` at module top
- **Imports**: Order: stdlib → third-party → local (`from src.X import Y` or relative imports within src/)

## Common Pitfalls

1. **Data leakage**: Never fit scalers/selectors on full dataset - only on CV train folds
2. **Optional deps**: Always check `*_AVAILABLE` flags before using torch/tabnet/shap/lime
3. **Memory**: Use `--test-mode` for large datasets, never load all CIC files without sampling
4. **Model profiles**: LR needs scaling/FS, trees don't - use correct profile from config
5. **Feature engineering**: Behavioral ratios must be computed BEFORE harmonization, not after
6. **Dataset source**: `dataset_source` must be added during early fusion and encoded as integer (0/1)
