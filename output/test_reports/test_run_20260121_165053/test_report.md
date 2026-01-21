# Test Execution Report

**Generated**: 2026-01-21 16:55:49

## Summary Statistics

- **Total Tests**: 38
- **Passed**: 34 (89.5%)
- **Failed**: 4 (10.5%)
- **Skipped**: 0 (0.0%)

## Test Results by Outcome

### ✅ Passed Tests

#### tests/test_algo_handling.py::test_get_algo_names_column

- **Duration**: 0.007s
- **Input**: N/A

**Input Matrices:**

- **df (DataFrame)**:
  - Shape: (9, 0)
  - Headers: 
  - Sample row: {}

**Validation Criteria (All Passed):**

1. ✅ Assertion: isinstance(algos
   - Condition: `isinstance(algos`

2. ✅ Assertion: len(algos) == 5
   - Condition: `len(algos) == 5`

3. ✅ Assertion: list(algos.values) == ['LR'
   - Condition: `list(algos.values) == ['LR'`

4. ✅ Assertion: algos.dtype == 'object'
   - Condition: `algos.dtype == 'object'`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 4/4 passed

#### tests/test_algo_handling.py::test_get_algo_names_index

- **Duration**: 0.006s
- **Input**: N/A

**Input Matrices:**

- **df (DataFrame)**:
  - Shape: (9, 0)
  - Headers: 
  - Sample row: {}

**Validation Criteria (All Passed):**

1. ✅ Assertion: isinstance(algos
   - Condition: `isinstance(algos`

2. ✅ Assertion: len(algos) == 3
   - Condition: `len(algos) == 3`

3. ✅ Assertion: list(algos.values) == ['LR'
   - Condition: `list(algos.values) == ['LR'`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 3/3 passed

#### tests/test_algo_handling.py::test_get_algo_names_raises

- **Duration**: 0.003s
- **Input**: N/A

**Input Matrices:**

- **df (DataFrame)**:
  - Shape: (9, 0)
  - Headers: 
  - Sample row: {}

- **Expected Output**: N/A
- **Success Reason**: Algorithm name handling (sanitization, column management) working correctly

#### tests/test_algo_handling.py::test_ensure_algo_column

- **Duration**: 0.005s
- **Input**: N/A

**Input Matrices:**

- **df1 (DataFrame)**:
  - Shape: (9, 0)
  - Headers: 
  - Sample row: {}

- **df2 (DataFrame)**:
  - Shape: (9, 0)
  - Headers: 
  - Sample row: {}

- **df4 (DataFrame)**:
  - Shape: (9, 0)
  - Headers: 
  - Sample row: {}

**Validation Criteria (All Passed):**

1. ✅ Assertion: result1 is not None
   - Condition: `result1 is not None`

2. ✅ Assertion: 'algo' in result1.columns
   - Condition: `'algo' in result1.columns`

3. ✅ Assertion: result1.equals(df1)
   - Condition: `result1.equals(df1)`

4. ✅ Assertion: result2 is not None
   - Condition: `result2 is not None`

5. ✅ Assertion: 'algo' in result2.columns
   - Condition: `'algo' in result2.columns`

6. ✅ Assertion: list(result2['algo'].values) == ['LR'
   - Condition: `list(result2['algo'].values) == ['LR'`

7. ✅ Assertion: result3 is None
   - Condition: `result3 is None`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 7/7 passed

#### tests/test_algo_handling.py::test_sanitize_algo_name

- **Duration**: 0.001s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: result == expected
   - Condition: `result == expected`

2. ✅ Assertion: result == label
   - Condition: `result == label`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 2/2 passed

#### tests/test_data_harmonization_small.py::test_analyze_feature_similarity_basic

- **Duration**: 0.008s
- **Input**: N/A

**Input Matrices:**

- **df1 (DataFrame)**:
  - Shape: (0, 2)
  - Headers: col_0, col_1
  - Sample row: {'col_0': 2.79, 'col_1': -9.5}

- **df2 (DataFrame)**:
  - Shape: (0, 2)
  - Headers: col_0, col_1
  - Sample row: {'col_0': 2.79, 'col_1': -9.5}

**Validation Criteria (All Passed):**

1. ✅ Assertion: result['compatible'] is True
   - Condition: `result['compatible'] is True`

2. ✅ Assertion: 'ks_pvalue' in result
   - Condition: `'ks_pvalue' in result`

3. ✅ Assertion: result['mean_diff'] < 0.5
   - Condition: `result['mean_diff'] < 0.5`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 3/3 passed

#### tests/test_data_harmonization_small.py::test_early_fusion_basic

- **Duration**: 0.016s
- **Input**: N/A

**Input Matrices:**

- **df_cic (DataFrame)**:
  - Shape: (1, 2)
  - Headers: col_0, col_1
  - Sample row: {'col_0': 2.79, 'col_1': -9.5}

- **df_ton (DataFrame)**:
  - Shape: (1, 2)
  - Headers: col_0, col_1
  - Sample row: {'col_0': 2.79, 'col_1': -9.5}

**Validation Criteria (All Passed):**

1. ✅ Assertion: len(fused) == 4
   - Condition: `len(fused) == 4`

2. ✅ Assertion: 'dataset_source' in fused.columns
   - Condition: `'dataset_source' in fused.columns`

3. ✅ Assertion: list(fused['dataset_source'].unique()) == [0
   - Condition: `list(fused['dataset_source'].unique()) == [0`

4. ✅ Assertion: 'f1' in validation
   - Condition: `'f1' in validation`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 4/4 passed

#### tests/test_data_harmonization_small.py::test_harmonize_features_labels

- **Duration**: 0.037s
- **Input**: N/A

**Input Matrices:**

- **df_cic (DataFrame)**:
  - Shape: (1, 2)
  - Headers: col_0, col_1
  - Sample row: {'col_0': 2.79, 'col_1': -9.5}

- **df_ton (DataFrame)**:
  - Shape: (1, 2)
  - Headers: col_0, col_1
  - Sample row: {'col_0': 2.79, 'col_1': -9.5}

**Validation Criteria (All Passed):**

1. ✅ Assertion: h_cic['label'].tolist() == [0
   - Condition: `h_cic['label'].tolist() == [0`

2. ✅ Assertion: h_ton['label'].tolist() == [0
   - Condition: `h_ton['label'].tolist() == [0`

3. ✅ Assertion: 'f1' in h_cic.columns
   - Condition: `'f1' in h_cic.columns`

4. ✅ Assertion: 'f1' in h_ton.columns
   - Condition: `'f1' in h_ton.columns`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 4/4 passed

#### tests/test_dataset_source_flag.py::test_dataset_source_flag

- **Duration**: 0.011s
- **Input**: N/A

**Input Matrices:**

- **df (DataFrame)**:
  - Shape: (1, 2)
  - Headers: col_0, col_1
  - Sample row: {'col_0': 2.79, 'col_1': -9.5}

**Validation Criteria (All Passed):**

1. ✅ Assertion: 'dataset_source' in X.columns
   - Condition: `'dataset_source' in X.columns`

2. ✅ Assertion: 'dataset_source' not in X_f.columns
   - Condition: `'dataset_source' not in X_f.columns`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 2/2 passed

#### tests/test_model_aware_profiles.py::test_model_aware_profiles

- **Duration**: 0.005s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: lr_profile["apply_scaling"] is True
   - Condition: `lr_profile["apply_scaling"] is True`

2. ✅ Assertion: lr_profile["apply_feature_selection"] is True
   - Condition: `lr_profile["apply_feature_selection"] is True`

3. ✅ Assertion: lr_profile["apply_resampling"] is True
   - Condition: `lr_profile["apply_resampling"] is True`

4. ✅ Assertion: lr_profile.get("use_class_weight"
   - Condition: `lr_profile.get("use_class_weight"`

5. ✅ Assertion: "feature_selection_k" in lr_profile
   - Condition: `"feature_selection_k" in lr_profile`

6. ✅ Assertion: 10 <= lr_profile["feature_selection_k"] <= 60
   - Condition: `10 <= lr_profile["feature_selection_k"] <= 60`

7. ✅ Assertion: cnn_profile["apply_scaling"] is True
   - Condition: `cnn_profile["apply_scaling"] is True`

8. ✅ Assertion: cnn_profile["apply_feature_selection"] is False
   - Condition: `cnn_profile["apply_feature_selection"] is False`

9. ✅ Assertion: cnn_profile["apply_resampling"] is True
   - Condition: `cnn_profile["apply_resampling"] is True`

10. ✅ Assertion: cnn_profile.get("cnn_reshape"
   - Condition: `cnn_profile.get("cnn_reshape"`

11. ✅ Assertion: tabnet_profile["apply_scaling"] is False
   - Condition: `tabnet_profile["apply_scaling"] is False`

12. ✅ Assertion: tabnet_profile["apply_feature_selection"] is False
   - Condition: `tabnet_profile["apply_feature_selection"] is False`

13. ✅ Assertion: tabnet_profile["apply_resampling"] is False
   - Condition: `tabnet_profile["apply_resampling"] is False`

14. ✅ Assertion: tabnet_profile["use_class_weight"] is True
   - Condition: `tabnet_profile["use_class_weight"] is True`

15. ✅ Assertion: tabnet_profile.get("class_weight") == "balanced"
   - Condition: `tabnet_profile.get("class_weight") == "balanced"`

16. ✅ Assertion: tree_profile["apply_scaling"] is False
   - Condition: `tree_profile["apply_scaling"] is False`

17. ✅ Assertion: tree_profile["apply_feature_selection"] is False
   - Condition: `tree_profile["apply_feature_selection"] is False`

18. ✅ Assertion: tree_profile["apply_resampling"] is False
   - Condition: `tree_profile["apply_resampling"] is False`

19. ✅ Assertion: tree_profile["use_class_weight"] is True
   - Condition: `tree_profile["use_class_weight"] is True`

20. ✅ Assertion: tree_profile.get("class_weight") == "balanced"
   - Condition: `tree_profile.get("class_weight") == "balanced"`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 20/20 passed

#### tests/test_new_pipeline_components.py::test_feature_categorization

- **Duration**: 0.003s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: 'Flow_Basic_Stats' in categorized
   - Condition: `'Flow_Basic_Stats' in categorized`

2. ✅ Assertion: 'Flow Duration' in categorized['Flow_Basic_Stats']
   - Condition: `'Flow Duration' in categorized['Flow_Basic_Stats']`

3. ✅ Assertion: 'Flow_Identifiers' in categorized
   - Condition: `'Flow_Identifiers' in categorized`

4. ✅ Assertion: 'Source IP' in categorized['Flow_Identifiers']
   - Condition: `'Source IP' in categorized['Flow_Identifiers']`

5. ✅ Assertion: 'Flag_Counts' in categorized
   - Condition: `'Flag_Counts' in categorized`

6. ✅ Assertion: 'SYN Flag Count' in categorized['Flag_Counts']
   - Condition: `'SYN Flag Count' in categorized['Flag_Counts']`

7. ✅ Assertion: 'Other' in categorized
   - Condition: `'Other' in categorized`

8. ✅ Assertion: 'Unknown_Col' in categorized['Other']
   - Condition: `'Unknown_Col' in categorized['Other']`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 8/8 passed

#### tests/test_new_pipeline_components.py::test_category_scores

- **Duration**: 0.002s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: 'performance' in scores
   - Condition: `'performance' in scores`

2. ✅ Assertion: 'explainability' in scores
   - Condition: `'explainability' in scores`

3. ✅ Assertion: 'resources' in scores
   - Condition: `'resources' in scores`

4. ✅ Assertion: scores['explainability'] > 0
   - Condition: `scores['explainability'] > 0`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 4/4 passed

#### tests/test_new_pipeline_components.py::test_trainer_single

- **Duration**: 0.024s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: 'DT' in trainer.models
   - Condition: `'DT' in trainer.models`

2. ✅ Assertion: trainer.models['DT'] is not None
   - Condition: `trainer.models['DT'] is not None`

3. ✅ Assertion: 'DT' in trainer.training_times
   - Condition: `'DT' in trainer.training_times`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 3/3 passed

#### tests/test_new_pipeline_components.py::test_validator_single

- **Duration**: 0.660s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: 'DT' in validator.best_params
   - Condition: `'DT' in validator.best_params`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 1/1 passed

#### tests/test_new_pipeline_components.py::test_xai_manager_single

- **Duration**: 2.271s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: 'RF' in xai.results
   - Condition: `'RF' in xai.results`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 1/1 passed

#### tests/test_new_pipeline_components.py::test_tester_single

- **Duration**: 0.485s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: 'DT' in tester.test_results
   - Condition: `'DT' in tester.test_results`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 1/1 passed

#### tests/test_no_data_leakage.py::test_scaler_fit_only_on_train

- **Duration**: 0.138s
- **Input**: N/A

**Input Matrices:**

- **X_train (DataFrame)**:
  - Shape: (200, 50)
  - Headers: col_0, col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9, col_10, col_11, col_12, col_13, col_14...
  - Sample row: {'col_0': 2.79, 'col_1': -9.5, 'col_2': -4.5, 'col_3': -5.54, 'col_4': 4.73}

- **X_test (DataFrame)**:
  - Shape: (0, 50)
  - Headers: f'feature_{i}' for i in range(n_features)
  - Sample row: {"f'feature_{i}' for i in range(n_features)": 2.79}

**Validation Criteria (All Passed):**

1. ✅ Assertion: pipeline.is_fitted
   - Condition: `pipeline.is_fitted`

2. ✅ Assertion: pipeline.scaler is not None
   - Condition: `pipeline.scaler is not None`

3. ✅ Assertion: X_test_transformed.shape[0] == n_test
   - Condition: `X_test_transformed.shape[0] == n_test`

4. ✅ Assertion: X_test_transformed.shape[1] == n_features
   - Condition: `X_test_transformed.shape[1] == n_features`

5. ✅ Assertion: np.abs(test_scaled_mean.mean()) < 5.0
   - Condition: `np.abs(test_scaled_mean.mean()) < 5.0`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 5/5 passed

#### tests/test_no_data_leakage.py::test_feature_selector_fit_only_on_train

- **Duration**: 0.368s
- **Input**: N/A

**Input Matrices:**

- **X_train (DataFrame)**:
  - Shape: (200, 50)
  - Headers: col_0, col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9, col_10, col_11, col_12, col_13, col_14...
  - Sample row: {'col_0': 2.79, 'col_1': -9.5, 'col_2': -4.5, 'col_3': -5.54, 'col_4': 4.73}

- **X_test (DataFrame)**:
  - Shape: (200, 50)
  - Headers: f'feature_{i}' for i in range(n_features)
  - Sample row: {"f'feature_{i}' for i in range(n_features)": 2.79}

**Validation Criteria (All Passed):**

1. ✅ Assertion: pipeline.feature_selector is not None
   - Condition: `pipeline.feature_selector is not None`

2. ✅ Assertion: pipeline.selected_features is not None
   - Condition: `pipeline.selected_features is not None`

3. ✅ Assertion: len(pipeline.selected_features) == k_selected
   - Condition: `len(pipeline.selected_features) == k_selected`

4. ✅ Assertion: X_test_transformed.shape[1] == k_selected
   - Condition: `X_test_transformed.shape[1] == k_selected`

5. ✅ Assertion: X_test_transformed.shape[0] == n_test
   - Condition: `X_test_transformed.shape[0] == n_test`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 5/5 passed

#### tests/test_no_data_leakage.py::test_imputer_fit_only_on_train

- **Duration**: 0.139s
- **Input**: N/A

**Input Matrices:**

- **X_train (DataFrame)**:
  - Shape: (200, 50)
  - Headers: col_0, col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9, col_10, col_11, col_12, col_13, col_14...
  - Sample row: {'col_0': 2.79, 'col_1': -9.5, 'col_2': -4.5, 'col_3': -5.54, 'col_4': 4.73}

- **X_test (DataFrame)**:
  - Shape: (200, 50)
  - Headers: f'feature_{i}' for i in range(n_features)
  - Sample row: {"f'feature_{i}' for i in range(n_features)": 2.79}

**Validation Criteria (All Passed):**

1. ✅ Assertion: pipeline.imputer is not None
   - Condition: `pipeline.imputer is not None`

2. ✅ Assertion: not np.isnan(X_test_transformed).any()
   - Condition: `not np.isnan(X_test_transformed).any()`

3. ✅ Assertion: np.abs(test_imputed_values.mean() - train_median_feature0) < 30.0
   - Condition: `np.abs(test_imputed_values.mean() - train_median_feature0) < 30.0`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 3/3 passed

#### tests/test_no_data_leakage.py::test_transform_test_no_fitting

- **Duration**: 0.126s
- **Input**: N/A

**Input Matrices:**

- **X_train (DataFrame)**:
  - Shape: (200, 50)
  - Headers: col_0, col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9, col_10, col_11, col_12, col_13, col_14...
  - Sample row: {'col_0': 2.79, 'col_1': -9.5, 'col_2': -4.5, 'col_3': -5.54, 'col_4': 4.73}

- **X_test (DataFrame)**:
  - Shape: (200, 50)
  - Headers: f'feature_{i}' for i in range(n_features)
  - Sample row: {"f'feature_{i}' for i in range(n_features)": 2.79}

**Validation Criteria (All Passed):**

1. ✅ Assertion: np.array_equal(pipeline.scaler.center_
   - Condition: `np.array_equal(pipeline.scaler.center_`

2. ✅ Assertion: np.array_equal(pipeline.scaler.scale_
   - Condition: `np.array_equal(pipeline.scaler.scale_`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 2/2 passed

#### tests/test_no_smote_leakage_phase3.py::test_no_smote_leakage_phase3

- **Duration**: 0.301s
- **Input**: N/A

**Input Matrices:**

- **X (DataFrame)**:
  - Shape: (100, 10)
  - Headers: f'f{i}' for i in range(10)
  - Sample row: {"f'f{i}' for i in range(10)": 2.79}

**Validation Criteria (All Passed):**

1. ✅ Assertion: "Applying SMOTE before splitting" not in caplog.text
   - Condition: `"Applying SMOTE before splitting" not in caplog.text`

2. ✅ Assertion: counts[0] == counts[1]
   - Condition: `counts[0] == counts[1]`

3. ✅ Assertion: counts[0] > 10 # Should be 90
   - Condition: `counts[0] > 10 # Should be 90`

4. ✅ Assertion: pipeline.smote is not None
   - Condition: `pipeline.smote is not None`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 4/4 passed

#### tests/test_performance_and_ram.py::test_ram_optimization_during_loading

- **Duration**: 0.368s
- **Input**: N/A

**Input Matrices:**

- **df (DataFrame)**:
  - Shape: (100, 10)
  - Headers: col_0, col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9
  - Sample row: {'col_0': 2.79, 'col_1': -9.5, 'col_2': -4.5, 'col_3': -5.54, 'col_4': 4.73}

**Validation Criteria (All Passed):**

1. ✅ Assertion: optimized_mem < initial_mem
   - Condition: `optimized_mem < initial_mem`

2. ✅ Assertion: df_opt['float_col'].dtype == 'float32'
   - Condition: `df_opt['float_col'].dtype == 'float32'`

3. ✅ Assertion: df_opt['int_col'].dtype in ['int16'
   - Condition: `df_opt['int_col'].dtype in ['int16'`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 3/3 passed

#### tests/test_performance_and_ram.py::test_multi_threading_config

- **Duration**: 0.003s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: rf.n_jobs == -1
   - Condition: `rf.n_jobs == -1`

2. ✅ Assertion: lr.n_jobs == -1
   - Condition: `lr.n_jobs == -1`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 2/2 passed

#### tests/test_performance_and_ram.py::test_system_monitor_ram_check

- **Duration**: 0.004s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: 'used_percent' in mem_info
   - Condition: `'used_percent' in mem_info`

2. ✅ Assertion: 'process_mem_mb' in mem_info
   - Condition: `'process_mem_mb' in mem_info`

3. ✅ Assertion: mem_info['process_mem_mb'] > 0
   - Condition: `mem_info['process_mem_mb'] > 0`

4. ✅ Assertion: isinstance(is_safe
   - Condition: `isinstance(is_safe`

5. ✅ Assertion: isinstance(msg
   - Condition: `isinstance(msg`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 5/5 passed

#### tests/test_phase2_outputs.py::test_phase2_outputs

- **Duration**: 0.075s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: output_paths["preprocessed_data"].exists()
   - Condition: `output_paths["preprocessed_data"].exists()`

2. ✅ Assertion: output_paths["feature_names"].exists()
   - Condition: `output_paths["feature_names"].exists()`

3. ✅ Assertion: output_paths["summary"].exists()
   - Condition: `output_paths["summary"].exists()`

4. ✅ Assertion: "dataset_source" in df.columns
   - Condition: `"dataset_source" in df.columns`

5. ✅ Assertion: df["dataset_source"].isin([0
   - Condition: `df["dataset_source"].isin([0`

6. ✅ Assertion: "label" in df.columns
   - Condition: `"label" in df.columns`

7. ✅ Assertion: "feature_names" in feature_data
   - Condition: `"feature_names" in feature_data`

8. ✅ Assertion: "dataset_source" in feature_data["feature_names"] or "dataset_source" not in df.
   - Condition: `"dataset_source" in feature_data["feature_names"] or "dataset_source" not in df.drop(columns=["label"]).columns`

9. ✅ Assertion: "Phase 2: Apply Best Configuration" in summary_content
   - Condition: `"Phase 2: Apply Best Configuration" in summary_content`

10. ✅ Assertion: "dataset_source" in summary_content or "Dataset Source Distribution" in summary_
   - Condition: `"dataset_source" in summary_content or "Dataset Source Distribution" in summary_content`

11. ✅ Assertion: "Stateless preprocessing only" in summary_content
   - Condition: `"Stateless preprocessing only" in summary_content`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 11/11 passed

#### tests/test_phase3_cnn_tabnet.py::test_phase3_cnn_tabnet_profiles

- **Duration**: 0.003s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: 'cnn_profile' in config.preprocessing_profiles
   - Condition: `'cnn_profile' in config.preprocessing_profiles`

2. ✅ Assertion: cnn_profile.get('apply_scaling') is True
   - Condition: `cnn_profile.get('apply_scaling') is True`

3. ✅ Assertion: cnn_profile.get('apply_feature_selection') is False
   - Condition: `cnn_profile.get('apply_feature_selection') is False`

4. ✅ Assertion: cnn_profile.get('cnn_reshape') is True
   - Condition: `cnn_profile.get('cnn_reshape') is True`

5. ✅ Assertion: cnn_profile.get('apply_resampling') is True
   - Condition: `cnn_profile.get('apply_resampling') is True`

6. ✅ Assertion: 'tabnet_profile' in config.preprocessing_profiles
   - Condition: `'tabnet_profile' in config.preprocessing_profiles`

7. ✅ Assertion: tabnet_profile.get('apply_scaling') is False
   - Condition: `tabnet_profile.get('apply_scaling') is False`

8. ✅ Assertion: tabnet_profile.get('apply_feature_selection') is False
   - Condition: `tabnet_profile.get('apply_feature_selection') is False`

9. ✅ Assertion: tabnet_profile.get('use_class_weight') is True
   - Condition: `tabnet_profile.get('use_class_weight') is True`

10. ✅ Assertion: tabnet_profile.get('class_weight') == 'balanced'
   - Condition: `tabnet_profile.get('class_weight') == 'balanced'`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 10/10 passed

#### tests/test_phase3_cnn_tabnet.py::test_phase3_model_names

- **Duration**: 0.002s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: 'cnn' in algos_lower or 'CNN' in config.phase3_algorithms
   - Condition: `'cnn' in algos_lower or 'CNN' in config.phase3_algorithms`

2. ✅ Assertion: 'tabnet' in algos_lower or 'TabNet' in config.phase3_algorithms
   - Condition: `'tabnet' in algos_lower or 'TabNet' in config.phase3_algorithms`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 2/2 passed

#### tests/test_phase3_cnn_tabnet.py::test_phase3_metrics_df_format

- **Duration**: 0.007s
- **Input**: N/A

**Input Matrices:**

- **metrics_df (DataFrame)**:
  - Shape: (9, 0)
  - Headers: 
  - Sample row: {}

**Validation Criteria (All Passed):**

1. ✅ Assertion: 'algo' in metrics_df_ensured.columns
   - Condition: `'algo' in metrics_df_ensured.columns`

2. ✅ Assertion: list(algos.values) == ['LR'
   - Condition: `list(algos.values) == ['LR'`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 2/2 passed

#### tests/test_preprocessing_pipeline.py::test_sanitize_numeric_values_removes_inf_and_clips

- **Duration**: 0.069s
- **Input**: N/A

**Input Matrices:**

- **X_df (DataFrame)**:
  - Shape: (4, 2)
  - Headers: f"feature_{i}" for i in range(n_features)
  - Sample row: {'f"feature_{i}" for i in range(n_features)': 2.79}

**Validation Criteria (All Passed):**

1. ✅ Assertion: X_sanitized.shape == X_df.shape
   - Condition: `X_sanitized.shape == X_df.shape`

2. ✅ Assertion: list(X_sanitized.columns) == list(X_df.columns)
   - Condition: `list(X_sanitized.columns) == list(X_df.columns)`

3. ✅ Assertion: inf_count == 0
   - Condition: `inf_count == 0`

4. ✅ Assertion: X_sanitized.loc[4
   - Condition: `X_sanitized.loc[4`

5. ✅ Assertion: X_sanitized.loc[5
   - Condition: `X_sanitized.loc[5`

6. ✅ Assertion: finite_sanitized.min() >= q_low
   - Condition: `finite_sanitized.min() >= q_low`

7. ✅ Assertion: finite_sanitized.max() <= q_high
   - Condition: `finite_sanitized.max() <= q_high`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 7/7 passed

#### tests/test_preprocessing_pipeline.py::test_sanitize_numeric_values_replace_inf_with_max

- **Duration**: 0.016s
- **Input**: N/A

**Input Matrices:**

- **X_df (DataFrame)**:
  - Shape: (100, 3)
  - Headers: feature_0, feature_1, feature_2
  - Sample row: {'feature_0': 2.79, 'feature_1': -9.5, 'feature_2': -4.5}

**Validation Criteria (All Passed):**

1. ✅ Assertion: inf_count == 0
   - Condition: `inf_count == 0`

2. ✅ Assertion: X_sanitized.loc[0
   - Condition: `X_sanitized.loc[0`

3. ✅ Assertion: X_sanitized.loc[1
   - Condition: `X_sanitized.loc[1`

4. ✅ Assertion: X_sanitized.loc[2
   - Condition: `X_sanitized.loc[2`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 4/4 passed

#### tests/test_preprocessing_pipeline.py::test_sanitize_numeric_values_abs_cap

- **Duration**: 0.015s
- **Input**: N/A

**Input Matrices:**

- **X_df (DataFrame)**:
  - Shape: (50, 3)
  - Headers: feature_0, feature_1, feature_2
  - Sample row: {'feature_0': 2.79, 'feature_1': -9.5, 'feature_2': -4.5}

**Validation Criteria (All Passed):**

1. ✅ Assertion: X_sanitized.values.min() >= -abs_cap
   - Condition: `X_sanitized.values.min() >= -abs_cap`

2. ✅ Assertion: X_sanitized.values.max() <= abs_cap
   - Condition: `X_sanitized.values.max() <= abs_cap`

3. ✅ Assertion: X_sanitized.loc[0
   - Condition: `X_sanitized.loc[0`

4. ✅ Assertion: X_sanitized.loc[1
   - Condition: `X_sanitized.loc[1`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 4/4 passed

#### tests/test_preprocessing_pipeline.py::test_sanitize_numeric_values_integration_with_clean_data

- **Duration**: 0.058s
- **Input**: N/A

**Input Matrices:**

- **X_df (DataFrame)**:
  - Shape: (100, 5)
  - Headers: f"feature_{i}" for i in range(5)
  - Sample row: {'f"feature_{i}" for i in range(5)': 2.79}

**Validation Criteria (All Passed):**

1. ✅ Assertion: inf_count == 0
   - Condition: `inf_count == 0`

2. ✅ Assertion: np.abs(X_cleaned.loc[2
   - Condition: `np.abs(X_cleaned.loc[2`

3. ✅ Assertion: X_cleaned.shape[0] == X_df.shape[0]
   - Condition: `X_cleaned.shape[0] == X_df.shape[0]`

4. ✅ Assertion: X_cleaned.shape[1] == X_df.shape[1]
   - Condition: `X_cleaned.shape[1] == X_df.shape[1]`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 4/4 passed

#### tests/test_preprocessing_pipeline.py::test_transform_test_requires_fitted_pipeline

- **Duration**: 0.088s
- **Input**: N/A

**Input Matrices:**

- **X_test_df (DataFrame)**:
  - Shape: (10, 5)
  - Headers: col_0, col_1, col_2, col_3, col_4
  - Sample row: {'col_0': 2.79, 'col_1': -9.5, 'col_2': -4.5, 'col_3': -5.54, 'col_4': 4.73}

- **X_train (DataFrame)**:
  - Shape: (20, 5)
  - Headers: col_0, col_1, col_2, col_3, col_4
  - Sample row: {'col_0': 2.79, 'col_1': -9.5, 'col_2': -4.5, 'col_3': -5.54, 'col_4': 4.73}

- **X_test_fitted (DataFrame)**:
  - Shape: (10, 5)
  - Headers: col_0, col_1, col_2, col_3, col_4
  - Sample row: {'col_0': 2.79, 'col_1': -9.5, 'col_2': -4.5, 'col_3': -5.54, 'col_4': 4.73}

**Validation Criteria (All Passed):**

1. ✅ Assertion: not pipeline.is_fitted
   - Condition: `not pipeline.is_fitted`

2. ✅ Assertion: pipeline.is_fitted
   - Condition: `pipeline.is_fitted`

3. ✅ Assertion: isinstance(result
   - Condition: `isinstance(result`

4. ✅ Assertion: result.shape[0] == 10
   - Condition: `result.shape[0] == 10`

5. ✅ Assertion: result.shape[1] == 5
   - Condition: `result.shape[1] == 5`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 5/5 passed

#### tests/test_test_transform_is_fold_fitted.py::test_test_transform_is_fold_fitted

- **Duration**: 0.145s
- **Input**: N/A

**Input Matrices:**

- **X_train (DataFrame)**:
  - Shape: (10, 10)
  - Headers: col_0, col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9
  - Sample row: {'col_0': 2.79, 'col_1': -9.5, 'col_2': -4.5, 'col_3': -5.54, 'col_4': 4.73}

- **X_test (DataFrame)**:
  - Shape: (10, 10)
  - Headers: col_0, col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9
  - Sample row: {'col_0': 2.79, 'col_1': -9.5, 'col_2': -4.5, 'col_3': -5.54, 'col_4': 4.73}

**Validation Criteria (All Passed):**

1. ✅ Assertion: not np.isnan(X_test_prep).any()
   - Condition: `not np.isnan(X_test_prep).any()`

2. ✅ Assertion: np.isfinite(X_test_prep).all()
   - Condition: `np.isfinite(X_test_prep).all()`

3. ✅ Assertion: not np.isnan(X_test_prep_t).any()
   - Condition: `not np.isnan(X_test_prep_t).any()`

4. ✅ Assertion: X_test_prep_t[0
   - Condition: `X_test_prep_t[0`

5. ✅ Assertion: X_test_prep_t[1
   - Condition: `X_test_prep_t[1`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 5/5 passed

### ❌ Failed Tests

#### tests/test_cnn.py::test_cnn_pipeline_full_flow

- **Duration**: 125.707s
- **Input**: N/A
**Validation Criteria:**

1. ❌ Assertion: 1 in results
   - Condition: `1 in results`

2. ❌ Assertion: 2 in results
   - Condition: `2 in results`

3. ❌ Assertion: 3 in results
   - Condition: `3 in results`

4. ❌ Assertion: 5 in results
   - Condition: `5 in results`

5. ❌ Assertion: runner.best_config is not None
   - Condition: `runner.best_config is not None`

6. ❌ Assertion: (test_results_dir / 'phase1_config_search' / 'best_config.json').exists()
   - Condition: `(test_results_dir / 'phase1_config_search' / 'best_config.json').exists()`

7. ❌ Assertion: algo in found_algos
   - Condition: `algo in found_algos`

- **Failure Reason**: Assertion failed - Expected condition not met
- **Error Message**: `tests/test_cnn.py:99: in test_cnn_pipeline_full_flow
    assert algo in found_algos, f"{algo} missing from evaluation. Found: {found_algos}"
E   AssertionError: LR missing from evaluation. Found: ['NO_MODELS_EVALUATED']
E   assert 'LR' in ['NO_MODELS_EVALUATED']`

```
tests/test_cnn.py:99: in test_cnn_pipeline_full_flow
    assert algo in found_algos, f"{algo} missing from evaluation. Found: {found_algos}"
E   AssertionError: LR missing from evaluation. Found: ['NO_MODELS_EVALUATED']
E   assert 'LR' in ['NO_MODELS_EVALUATED']
```

#### tests/test_no_global_fit_regression.py::test_no_global_fit_regression

- **Duration**: 0.083s
- **Input**: N/A

**Input Matrices:**

- **df (DataFrame)**:
  - Shape: (20, 5)
  - Headers: f'f{i}' for i in range(5)
  - Sample row: {"f'f{i}' for i in range(5)": 2.79}

- **return_value (DataFrame)**:
  - Shape: (100, 10)
  - Headers: col_0, col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9
  - Sample row: {'col_0': 2.79, 'col_1': -9.5, 'col_2': -4.5, 'col_3': -5.54, 'col_4': 4.73}

**Validation Criteria:**

1. ❌ Assertion: MockPipeline.call_count == 2
   - Condition: `MockPipeline.call_count == 2`

- **Failure Reason**: Invalid value provided
- **Error Message**: `tests/test_no_global_fit_regression.py:60: in test_no_global_fit_regression
    evaluator.run()
src/phases/phase3_evaluation.py:118: in run
    raise ValueError("No algorithms available for evaluation. Check dependencies and config.")
E   ValueError: No algorithms available for evaluation. Check dependencies and config.`

```
tests/test_no_global_fit_regression.py:60: in test_no_global_fit_regression
    evaluator.run()
src/phases/phase3_evaluation.py:118: in run
    raise ValueError("No algorithms available for evaluation. Check dependencies and config.")
E   ValueError: No algorithms available for evaluation. Check dependencies and config.
```

#### tests/test_phase3_synthetic.py::test_phase3_synthetic_produces_outputs

- **Duration**: 0.087s
- **Input**: N/A
**Validation Criteria:**

1. ❌ Assertion: results_file.exists()
   - Condition: `results_file.exists()`

2. ❌ Assertion: len(results_df) > 0
   - Condition: `len(results_df) > 0`

3. ❌ Assertion: "model_name" in results_df.columns
   - Condition: `"model_name" in results_df.columns`

4. ❌ Assertion: "f1_score" in results_df.columns
   - Condition: `"f1_score" in results_df.columns`

5. ❌ Assertion: len(dimension_scores_df) > 0
   - Condition: `len(dimension_scores_df) > 0`

6. ❌ Assertion: "model_name" in dimension_scores_df.columns
   - Condition: `"model_name" in dimension_scores_df.columns`

7. ❌ Assertion: len(report_files) > 0
   - Condition: `len(report_files) > 0`

- **Failure Reason**: Invalid value provided
- **Error Message**: `tests/test_phase3_synthetic.py:51: in test_phase3_synthetic_produces_outputs
    result = phase3.run()
             ^^^^^^^^^^^^
src/phases/phase3_evaluation.py:118: in run
    raise ValueError("No algorithms available for evaluation. Check dependencies and config.")
E   ValueError: No algorithms available for evaluation. Check dependencies and config.`

```
tests/test_phase3_synthetic.py:51: in test_phase3_synthetic_produces_outputs
    result = phase3.run()
             ^^^^^^^^^^^^
src/phases/phase3_evaluation.py:118: in run
    raise ValueError("No algorithms available for evaluation. Check dependencies and config.")
E   ValueError: No algorithms available for evaluation. Check dependencies and config.
```

#### tests/test_tabnet.py::test_tabnet_pipeline_full_flow

- **Duration**: 155.384s
- **Input**: N/A
**Validation Criteria:**

1. ❌ Assertion: 1 in results
   - Condition: `1 in results`

2. ❌ Assertion: 2 in results
   - Condition: `2 in results`

3. ❌ Assertion: 3 in results
   - Condition: `3 in results`

4. ❌ Assertion: 5 in results
   - Condition: `5 in results`

5. ❌ Assertion: runner.best_config is not None
   - Condition: `runner.best_config is not None`

6. ❌ Assertion: (test_results_dir / 'phase1_config_search' / 'best_config.json').exists()
   - Condition: `(test_results_dir / 'phase1_config_search' / 'best_config.json').exists()`

7. ❌ Assertion: algo in found_algos
   - Condition: `algo in found_algos`

- **Failure Reason**: Assertion failed - Expected condition not met
- **Error Message**: `tests/test_tabnet.py:99: in test_tabnet_pipeline_full_flow
    assert algo in found_algos, f"{algo} missing from evaluation. Found: {found_algos}"
E   AssertionError: LR missing from evaluation. Found: ['NO_MODELS_EVALUATED']
E   assert 'LR' in ['NO_MODELS_EVALUATED']`

```
tests/test_tabnet.py:99: in test_tabnet_pipeline_full_flow
    assert algo in found_algos, f"{algo} missing from evaluation. Found: {found_algos}"
E   AssertionError: LR missing from evaluation. Found: ['NO_MODELS_EVALUATED']
E   assert 'LR' in ['NO_MODELS_EVALUATED']
```

