# Phase 2: Apply Best Configuration - Summary

**Date**: 2026-01-20 20:53:12

## Dataset Information
- **Total Rows**: 36,970
- **Total Features**: 1
- **Total Columns** (including label): 2

## Dataset Source Distribution
- **CIC-DDoS2019** (dataset_source=0): 36,620 rows
- **TON_IoT** (dataset_source=1): 350 rows

**Mapping**: `dataset_source` encoding: 0=CIC-DDoS2019, 1=TON_IoT

## Label Distribution
- **Class 0**: 36,870 rows (99.73%)
- **Class 1**: 100 rows (0.27%)

## Preprocessing Steps Applied (Phase 2)

**Stateless preprocessing only**:
- ✅ Data cleaning (NaN/Infinity removal, median imputation)
- ✅ Feature encoding (categorical features)
- ❌ Feature selection (NOT applied - will be done in Phase 3 per fold)
- ❌ Scaling (NOT applied - will be done in Phase 3 per fold)
- ❌ SMOTE resampling (NOT applied - will be done in Phase 3 per fold)

**Note**: Fit-dependent steps (scaling, feature selection, SMOTE) are applied in Phase 3 per fold to ensure zero data leakage.

## Best Configuration (from Phase 1)

```json
{
  "apply_cleaning": true,
  "apply_encoding": true,
  "apply_feature_selection": true,
  "feature_selection_k": 10,
  "apply_scaling": true,
  "scaling_method": "RobustScaler",
  "apply_resampling": true,
  "resampling_method": "SMOTE"
}
```

*Note: This configuration is documented for traceability. Only stateless steps are applied in Phase 2.*

## Memory Footprint
- **Estimated memory usage**: 0.56 MB

## Output Files
- `best_preprocessed.parquet` (or `.csv.gz`): Preprocessed dataset
- `feature_names.json`: List of feature names
- `phase2_summary.md`: This summary file
