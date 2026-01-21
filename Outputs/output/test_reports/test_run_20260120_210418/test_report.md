# Test Execution Report

**Generated**: 2026-01-20 21:08:00

## Summary Statistics

- **Total Tests**: 26
- **Passed**: 26 (100.0%)
- **Failed**: 0 (0.0%)
- **Skipped**: 0 (0.0%)

## Test Results by Outcome

### ✅ Passed Tests

#### tests/test_algo_handling.py::test_get_algo_names_column

- **Duration**: 0.003s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: isinstance(algos, pd.Series)
   - Condition: `isinstance(algos, pd.Series)`

2. ✅ len(algos) compares to 5
   - Condition: `len(algos) == 5`

3. ✅ list(algos.values) compares to ['LR', 'DT', 'RF', 'CNN', 'TabNet']
   - Condition: `list(algos.values) == ['LR', 'DT', 'RF', 'CNN', 'TabNet']`

4. ✅ algos.dtype compares to 'object'
   - Condition: `algos.dtype == 'object'`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 4/4 passed

#### tests/test_algo_handling.py::test_get_algo_names_index

- **Duration**: 0.003s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: isinstance(algos, pd.Series)
   - Condition: `isinstance(algos, pd.Series)`

2. ✅ len(algos) compares to 3
   - Condition: `len(algos) == 3`

3. ✅ list(algos.values) compares to ['LR', 'DT', 'RF']
   - Condition: `list(algos.values) == ['LR', 'DT', 'RF']`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 3/3 passed

#### tests/test_algo_handling.py::test_get_algo_names_raises

- **Duration**: 0.002s
- **Input**: N/A
- **Expected Output**: N/A
- **Success Reason**: Test passed - All assertions satisfied

#### tests/test_algo_handling.py::test_ensure_algo_column

- **Duration**: 0.003s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ 'algo' compares to result1.columns
   - Condition: `'algo' in result1.columns`

2. ✅ Assertion: result1.equals(df1)
   - Condition: `result1.equals(df1)`

3. ✅ 'algo' compares to result2.columns
   - Condition: `'algo' in result2.columns`

4. ✅ list(result2['algo'].values) compares to ['LR', 'DT']
   - Condition: `list(result2['algo'].values) == ['LR', 'DT']`

5. ✅ result3 compares to None
   - Condition: `result3 is None`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 5/5 passed

#### tests/test_algo_handling.py::test_sanitize_algo_name

- **Duration**: 0.001s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ result compares to expected
   - Condition: `result == expected`

2. ✅ result compares to label
   - Condition: `result == label`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 2/2 passed

#### tests/test_cnn.py::test_cnn_pipeline_full_flow

- **Duration**: 99.769s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ 1 compares to results
   - Condition: `1 in results`

2. ✅ 2 compares to results
   - Condition: `2 in results`

3. ✅ 3 compares to results
   - Condition: `3 in results`

4. ✅ 5 compares to results
   - Condition: `5 in results`

5. ✅ runner.best_config compares to None
   - Condition: `runner.best_config is not None`

6. ✅ Assertion: (test_results_dir / 'phase1_config_search' / 'best_config.json').exists()
   - Condition: `(test_results_dir / 'phase1_config_search' / 'best_config.json').exists()`

7. ✅ algo compares to found_algos
   - Condition: `algo in found_algos`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 7/7 passed

#### tests/test_dataset_source_flag.py::test_dataset_source_flag

- **Duration**: 0.028s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ 'dataset_source' compares to X.columns
   - Condition: `'dataset_source' in X.columns`

2. ✅ 'dataset_source' compares to X_f.columns
   - Condition: `'dataset_source' not in X_f.columns`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 2/2 passed

#### tests/test_model_aware_profiles.py::test_model_aware_profiles

- **Duration**: 0.011s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ lr_profile["apply_scaling"] compares to True
   - Condition: `lr_profile["apply_scaling"] is True`

2. ✅ lr_profile["apply_feature_selection"] compares to True
   - Condition: `lr_profile["apply_feature_selection"] is True`

3. ✅ lr_profile["apply_resampling"] compares to True
   - Condition: `lr_profile["apply_resampling"] is True`

4. ✅ lr_profile.get("use_class_weight", False) compares to False
   - Condition: `lr_profile.get("use_class_weight", False) is False`

5. ✅ "feature_selection_k" compares to lr_profile
   - Condition: `"feature_selection_k" in lr_profile`

6. ✅ Assertion: 10 <= lr_profile["feature_selection_k"] <= 60
   - Condition: `10 <= lr_profile["feature_selection_k"] <= 60`

7. ✅ cnn_profile["apply_scaling"] compares to True
   - Condition: `cnn_profile["apply_scaling"] is True`

8. ✅ cnn_profile["apply_feature_selection"] compares to False
   - Condition: `cnn_profile["apply_feature_selection"] is False`

9. ✅ cnn_profile["apply_resampling"] compares to True
   - Condition: `cnn_profile["apply_resampling"] is True`

10. ✅ cnn_profile.get("cnn_reshape", False) compares to True
   - Condition: `cnn_profile.get("cnn_reshape", False) is True`

11. ✅ tabnet_profile["apply_scaling"] compares to False
   - Condition: `tabnet_profile["apply_scaling"] is False`

12. ✅ tabnet_profile["apply_feature_selection"] compares to False
   - Condition: `tabnet_profile["apply_feature_selection"] is False`

13. ✅ tabnet_profile["apply_resampling"] compares to False
   - Condition: `tabnet_profile["apply_resampling"] is False`

14. ✅ tabnet_profile["use_class_weight"] compares to True
   - Condition: `tabnet_profile["use_class_weight"] is True`

15. ✅ tabnet_profile.get("class_weight") compares to "balanced"
   - Condition: `tabnet_profile.get("class_weight") == "balanced"`

16. ✅ tree_profile["apply_scaling"] compares to False
   - Condition: `tree_profile["apply_scaling"] is False`

17. ✅ tree_profile["apply_feature_selection"] compares to False
   - Condition: `tree_profile["apply_feature_selection"] is False`

18. ✅ tree_profile["apply_resampling"] compares to False
   - Condition: `tree_profile["apply_resampling"] is False`

19. ✅ tree_profile["use_class_weight"] compares to True
   - Condition: `tree_profile["use_class_weight"] is True`

20. ✅ tree_profile.get("class_weight") compares to "balanced"
   - Condition: `tree_profile.get("class_weight") == "balanced"`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 20/20 passed

#### tests/test_no_data_leakage.py::test_scaler_fit_only_on_train

- **Duration**: 0.143s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: pipeline.is_fitted
   - Condition: `pipeline.is_fitted`

2. ✅ pipeline.scaler compares to None
   - Condition: `pipeline.scaler is not None`

3. ✅ X_test_transformed.shape[0] compares to n_test
   - Condition: `X_test_transformed.shape[0] == n_test`

4. ✅ X_test_transformed.shape[1] compares to n_features
   - Condition: `X_test_transformed.shape[1] == n_features`

5. ✅ np.abs(test_scaled_mean.mean()) compares to 5.0
   - Condition: `np.abs(test_scaled_mean.mean()) < 5.0`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 5/5 passed

#### tests/test_no_data_leakage.py::test_feature_selector_fit_only_on_train

- **Duration**: 0.344s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ pipeline.feature_selector compares to None
   - Condition: `pipeline.feature_selector is not None`

2. ✅ pipeline.selected_features compares to None
   - Condition: `pipeline.selected_features is not None`

3. ✅ len(pipeline.selected_features) compares to k_selected
   - Condition: `len(pipeline.selected_features) == k_selected`

4. ✅ X_test_transformed.shape[1] compares to k_selected
   - Condition: `X_test_transformed.shape[1] == k_selected`

5. ✅ X_test_transformed.shape[0] compares to n_test
   - Condition: `X_test_transformed.shape[0] == n_test`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 5/5 passed

#### tests/test_no_data_leakage.py::test_imputer_fit_only_on_train

- **Duration**: 0.088s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ pipeline.imputer compares to None
   - Condition: `pipeline.imputer is not None`

2. ✅ Assertion: not np.isnan(X_test_transformed).any()
   - Condition: `not np.isnan(X_test_transformed).any()`

3. ✅ np.abs(test_imputed_values.mean() - train_median_feature0) compares to 30.0
   - Condition: `np.abs(test_imputed_values.mean() - train_median_feature0) < 30.0`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 3/3 passed

#### tests/test_no_data_leakage.py::test_transform_test_no_fitting

- **Duration**: 0.119s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: np.array_equal(pipeline.scaler.center_, scaler_center_before)
   - Condition: `np.array_equal(pipeline.scaler.center_, scaler_center_before)`

2. ✅ Assertion: np.array_equal(pipeline.scaler.scale_, scaler_scale_before)
   - Condition: `np.array_equal(pipeline.scaler.scale_, scaler_scale_before)`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 2/2 passed

#### tests/test_no_global_fit_regression.py::test_no_global_fit_regression

- **Duration**: 0.040s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ MockPipeline.call_count compares to 2
   - Condition: `MockPipeline.call_count == 2`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 1/1 passed

#### tests/test_no_smote_leakage_phase3.py::test_no_smote_leakage_phase3

- **Duration**: 0.168s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ "Applying SMOTE before splitting" compares to caplog.text
   - Condition: `"Applying SMOTE before splitting" not in caplog.text`

2. ✅ counts[0] compares to counts[1]
   - Condition: `counts[0] == counts[1]`

3. ✅ counts[0] compares to 10
   - Condition: `counts[0] > 10`

4. ✅ pipeline.smote compares to None
   - Condition: `pipeline.smote is not None`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 4/4 passed

#### tests/test_phase2_outputs.py::test_phase2_outputs

- **Duration**: 0.060s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: output_paths["preprocessed_data"].exists()
   - Condition: `output_paths["preprocessed_data"].exists()`

2. ✅ Assertion: output_paths["feature_names"].exists()
   - Condition: `output_paths["feature_names"].exists()`

3. ✅ Assertion: output_paths["summary"].exists()
   - Condition: `output_paths["summary"].exists()`

4. ✅ "dataset_source" compares to df.columns
   - Condition: `"dataset_source" in df.columns`

5. ✅ Assertion: df["dataset_source"].isin([0, 1]).all()
   - Condition: `df["dataset_source"].isin([0, 1]).all()`

6. ✅ "label" compares to df.columns
   - Condition: `"label" in df.columns`

7. ✅ "feature_names" compares to feature_data
   - Condition: `"feature_names" in feature_data`

8. ✅ Assertion: "dataset_source" in feature_data["feature_names"] or "dataset_source" not in df.drop(columns=["label
   - Condition: `"dataset_source" in feature_data["feature_names"] or "dataset_source" not in df.drop(columns=["label"]).columns`

9. ✅ "Phase 2: Apply Best Configuration" compares to summary_content
   - Condition: `"Phase 2: Apply Best Configuration" in summary_content`

10. ✅ Assertion: "dataset_source" in summary_content or "Dataset Source Distribution" in summary_content
   - Condition: `"dataset_source" in summary_content or "Dataset Source Distribution" in summary_content`

11. ✅ "Stateless preprocessing only" compares to summary_content
   - Condition: `"Stateless preprocessing only" in summary_content`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 11/11 passed

#### tests/test_phase3_cnn_tabnet.py::test_phase3_cnn_tabnet_profiles

- **Duration**: 0.002s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ 'cnn_profile' compares to config.preprocessing_profiles
   - Condition: `'cnn_profile' in config.preprocessing_profiles`

2. ✅ cnn_profile.get('apply_scaling') compares to True
   - Condition: `cnn_profile.get('apply_scaling') is True`

3. ✅ cnn_profile.get('apply_feature_selection') compares to False
   - Condition: `cnn_profile.get('apply_feature_selection') is False`

4. ✅ cnn_profile.get('cnn_reshape') compares to True
   - Condition: `cnn_profile.get('cnn_reshape') is True`

5. ✅ cnn_profile.get('apply_resampling') compares to True
   - Condition: `cnn_profile.get('apply_resampling') is True`

6. ✅ 'tabnet_profile' compares to config.preprocessing_profiles
   - Condition: `'tabnet_profile' in config.preprocessing_profiles`

7. ✅ tabnet_profile.get('apply_scaling') compares to False
   - Condition: `tabnet_profile.get('apply_scaling') is False`

8. ✅ tabnet_profile.get('apply_feature_selection') compares to False
   - Condition: `tabnet_profile.get('apply_feature_selection') is False`

9. ✅ tabnet_profile.get('use_class_weight') compares to True
   - Condition: `tabnet_profile.get('use_class_weight') is True`

10. ✅ tabnet_profile.get('class_weight') compares to 'balanced'
   - Condition: `tabnet_profile.get('class_weight') == 'balanced'`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 10/10 passed

#### tests/test_phase3_cnn_tabnet.py::test_phase3_model_names

- **Duration**: 0.001s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: 'cnn' in algos_lower or 'CNN' in config.phase3_algorithms
   - Condition: `'cnn' in algos_lower or 'CNN' in config.phase3_algorithms`

2. ✅ Assertion: 'tabnet' in algos_lower or 'TabNet' in config.phase3_algorithms
   - Condition: `'tabnet' in algos_lower or 'TabNet' in config.phase3_algorithms`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 2/2 passed

#### tests/test_phase3_cnn_tabnet.py::test_phase3_metrics_df_format

- **Duration**: 0.004s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ 'algo' compares to metrics_df_ensured.columns
   - Condition: `'algo' in metrics_df_ensured.columns`

2. ✅ list(algos.values) compares to ['LR', 'CNN', 'TabNet']
   - Condition: `list(algos.values) == ['LR', 'CNN', 'TabNet']`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 2/2 passed

#### tests/test_phase3_synthetic.py::test_phase3_synthetic_produces_outputs

- **Duration**: 5.049s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: results_file.exists()
   - Condition: `results_file.exists()`

2. ✅ len(results_df) compares to 0
   - Condition: `len(results_df) > 0`

3. ✅ "model_name" compares to results_df.columns
   - Condition: `"model_name" in results_df.columns`

4. ✅ "f1_score" compares to results_df.columns
   - Condition: `"f1_score" in results_df.columns`

5. ✅ len(dimension_scores_df) compares to 0
   - Condition: `len(dimension_scores_df) > 0`

6. ✅ "model_name" compares to dimension_scores_df.columns
   - Condition: `"model_name" in dimension_scores_df.columns`

7. ✅ len(report_files) compares to 0
   - Condition: `len(report_files) > 0`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 7/7 passed

#### tests/test_preprocessing_pipeline.py::test_sanitize_numeric_values_removes_inf_and_clips

- **Duration**: 0.016s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ X_sanitized.shape compares to X_df.shape
   - Condition: `X_sanitized.shape == X_df.shape`

2. ✅ list(X_sanitized.columns) compares to list(X_df.columns)
   - Condition: `list(X_sanitized.columns) == list(X_df.columns)`

3. ✅ inf_count compares to 0
   - Condition: `inf_count == 0`

4. ✅ X_sanitized.loc[4, 'feature_2'] compares to q_high_2
   - Condition: `X_sanitized.loc[4, 'feature_2'] <= q_high_2`

5. ✅ X_sanitized.loc[5, 'feature_2'] compares to q_low_2
   - Condition: `X_sanitized.loc[5, 'feature_2'] >= q_low_2`

6. ✅ finite_sanitized.min() compares to q_low
   - Condition: `finite_sanitized.min() >= q_low`

7. ✅ finite_sanitized.max() compares to q_high
   - Condition: `finite_sanitized.max() <= q_high`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 7/7 passed

#### tests/test_preprocessing_pipeline.py::test_sanitize_numeric_values_replace_inf_with_max

- **Duration**: 0.006s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ inf_count compares to 0
   - Condition: `inf_count == 0`

2. ✅ X_sanitized.loc[0, 'feature_0'] compares to expected_max_0
   - Condition: `X_sanitized.loc[0, 'feature_0'] == expected_max_0`

3. ✅ X_sanitized.loc[1, 'feature_0'] compares to expected_min_0
   - Condition: `X_sanitized.loc[1, 'feature_0'] == expected_min_0`

4. ✅ X_sanitized.loc[2, 'feature_1'] compares to expected_max_1
   - Condition: `X_sanitized.loc[2, 'feature_1'] == expected_max_1`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 4/4 passed

#### tests/test_preprocessing_pipeline.py::test_sanitize_numeric_values_abs_cap

- **Duration**: 0.004s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ X_sanitized.values.min() compares to -abs_cap
   - Condition: `X_sanitized.values.min() >= -abs_cap`

2. ✅ X_sanitized.values.max() compares to abs_cap
   - Condition: `X_sanitized.values.max() <= abs_cap`

3. ✅ X_sanitized.loc[0, 'feature_0'] compares to abs_cap
   - Condition: `X_sanitized.loc[0, 'feature_0'] == abs_cap`

4. ✅ X_sanitized.loc[1, 'feature_0'] compares to -abs_cap
   - Condition: `X_sanitized.loc[1, 'feature_0'] == -abs_cap`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 4/4 passed

#### tests/test_preprocessing_pipeline.py::test_sanitize_numeric_values_integration_with_clean_data

- **Duration**: 0.022s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ inf_count compares to 0
   - Condition: `inf_count == 0`

2. ✅ np.abs(X_cleaned.loc[2, 'feature_1']) compares to np.abs(q_high_1) * 2
   - Condition: `np.abs(X_cleaned.loc[2, 'feature_1']) <= np.abs(q_high_1) * 2`

3. ✅ X_cleaned.shape[0] compares to X_df.shape[0]
   - Condition: `X_cleaned.shape[0] == X_df.shape[0]`

4. ✅ X_cleaned.shape[1] compares to X_df.shape[1]
   - Condition: `X_cleaned.shape[1] == X_df.shape[1]`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 4/4 passed

#### tests/test_preprocessing_pipeline.py::test_transform_test_requires_fitted_pipeline

- **Duration**: 0.025s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: not pipeline.is_fitted
   - Condition: `not pipeline.is_fitted`

2. ✅ Assertion: pipeline.is_fitted
   - Condition: `pipeline.is_fitted`

3. ✅ Assertion: isinstance(result, np.ndarray)
   - Condition: `isinstance(result, np.ndarray)`

4. ✅ result.shape[0] compares to 10
   - Condition: `result.shape[0] == 10`

5. ✅ result.shape[1] compares to 5
   - Condition: `result.shape[1] == 5`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 5/5 passed

#### tests/test_tabnet.py::test_tabnet_pipeline_full_flow

- **Duration**: 110.466s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ 1 compares to results
   - Condition: `1 in results`

2. ✅ 2 compares to results
   - Condition: `2 in results`

3. ✅ 3 compares to results
   - Condition: `3 in results`

4. ✅ 5 compares to results
   - Condition: `5 in results`

5. ✅ runner.best_config compares to None
   - Condition: `runner.best_config is not None`

6. ✅ Assertion: (test_results_dir / 'phase1_config_search' / 'best_config.json').exists()
   - Condition: `(test_results_dir / 'phase1_config_search' / 'best_config.json').exists()`

7. ✅ algo compares to found_algos
   - Condition: `algo in found_algos`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 7/7 passed

#### tests/test_test_transform_is_fold_fitted.py::test_test_transform_is_fold_fitted

- **Duration**: 0.101s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: not np.isnan(X_test_prep).any()
   - Condition: `not np.isnan(X_test_prep).any()`

2. ✅ Assertion: np.isfinite(X_test_prep).all()
   - Condition: `np.isfinite(X_test_prep).all()`

3. ✅ Assertion: not np.isnan(X_test_prep_t).any()
   - Condition: `not np.isnan(X_test_prep_t).any()`

4. ✅ X_test_prep_t[0, 0] compares to 10.0
   - Condition: `X_test_prep_t[0, 0] == 10.0`

5. ✅ X_test_prep_t[1, 0] compares to 10.0
   - Condition: `X_test_prep_t[1, 0] == 10.0`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 5/5 passed

