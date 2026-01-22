# Test Execution Report

**Generated**: 2026-01-22 16:23:17

## Summary Statistics

- **Total Tests**: 14
- **Passed**: 14 (100.0%)
- **Failed**: 0 (0.0%)
- **Skipped**: 0 (0.0%)

## Test Results by Outcome

### ✅ Passed Tests

#### tests/test_contracts_validation.py::test_pipeline_config_invalid_algorithms

- **Duration**: 0.002s
- **Input**: N/A
- **Expected Output**: N/A
- **Success Reason**: Algorithm name handling (sanitization, column management) working correctly

#### tests/test_contracts_validation.py::test_pipeline_config_invalid_order

- **Duration**: 0.002s
- **Input**: N/A
- **Expected Output**: N/A
- **Success Reason**: Test passed: tests/test_contracts_validation.py::test_pipeline_config_invalid_order - All assertions satisfied

#### tests/test_contracts_validation.py::test_pipeline_config_valid

- **Duration**: 0.001s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: cfg.algorithms == ["LR"
   - Condition: `cfg.algorithms == ["LR"`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 1/1 passed

#### tests/test_dag_runner_dummy_tasks.py::test_dag_runner_execution

- **Duration**: 0.002s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: len(results) == 2
   - Condition: `len(results) == 2`

2. ✅ Assertion: results[0].task_name == "A"
   - Condition: `results[0].task_name == "A"`

3. ✅ Assertion: results[1].task_name == "B"
   - Condition: `results[1].task_name == "B"`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 3/3 passed

#### tests/test_event_bus_queue.py::test_event_bus_publish_subscribe

- **Duration**: 0.202s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: len(received) == 1
   - Condition: `len(received) == 1`

2. ✅ Assertion: received[0].type == "STATUS"
   - Condition: `received[0].type == "STATUS"`

3. ✅ Assertion: received[0].payload["message"] == "test"
   - Condition: `received[0].payload["message"] == "test"`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 3/3 passed

#### tests/test_feature_order_strict.py::test_feature_order_strict_transform

- **Duration**: 0.016s
- **Input**: N/A

**Input Matrices:**

- **df_train (DataFrame)**:
  - Shape: (1, 2)
  - Headers: col_0, col_1
  - Sample row: {'col_0': 2.79, 'col_1': -9.5}

- **df_test_missing (DataFrame)**:
  - Shape: (1, 2)
  - Headers: col_0, col_1
  - Sample row: {'col_0': 2.79, 'col_1': -9.5}

- **df_test_ok (DataFrame)**:
  - Shape: (5, 6)
  - Headers: col_0, col_1, col_2, col_3, col_4, col_5
  - Sample row: {'col_0': 2.79, 'col_1': -9.5, 'col_2': -4.5, 'col_3': -5.54, 'col_4': 4.73}

**Validation Criteria (All Passed):**

1. ✅ Assertion: X_trans.shape == (2
   - Condition: `X_trans.shape == (2`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 1/1 passed

#### tests/test_label_rules_cic_ton.py::test_cic_label_rule

- **Duration**: 0.041s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: df["y"].to_list() == [0
   - Condition: `df["y"].to_list() == [0`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 1/1 passed

#### tests/test_label_rules_cic_ton.py::test_ton_label_rule

- **Duration**: 0.015s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: df["y"].to_list() == [0
   - Condition: `df["y"].to_list() == [0`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 1/1 passed

#### tests/test_pipeline_graph_generation.py::test_pipeline_graph_generation

- **Duration**: 0.002s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: "graph TD" in mermaid
   - Condition: `"graph TD" in mermaid`

2. ✅ Assertion: "T00_InitRun" in mermaid
   - Condition: `"T00_InitRun" in mermaid`

3. ✅ Assertion: "T17_Evaluate" in mermaid
   - Condition: `"T17_Evaluate" in mermaid`

4. ✅ Assertion: len(order) == 16
   - Condition: `len(order) == 16`

5. ✅ Assertion: order[0] == "T00_InitRun"
   - Condition: `order[0] == "T00_InitRun"`

6. ✅ Assertion: order[-1] == "T17_Evaluate"
   - Condition: `order[-1] == "T17_Evaluate"`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 6/6 passed

#### tests/test_profiler_smoke.py::test_profiler_smoke

- **Duration**: 0.012s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: profile.n_rows == 4
   - Condition: `profile.n_rows == 4`

2. ✅ Assertion: profile.n_cols == 3
   - Condition: `profile.n_cols == 3`

3. ✅ Assertion: "feat1" in profile.numeric_summary
   - Condition: `"feat1" in profile.numeric_summary`

4. ✅ Assertion: "feat2" in profile.top_categories
   - Condition: `"feat2" in profile.top_categories`

5. ✅ Assertion: profile.label_balance == {"0": 2
   - Condition: `profile.label_balance == {"0": 2`

6. ✅ Assertion: profile.missing_rate["feat1"] == 0.25
   - Condition: `profile.missing_rate["feat1"] == 0.25`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 6/6 passed

#### tests/test_reducer_state_updates.py::test_reducer_task_started

- **Duration**: 0.003s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: "TaskA" in new_state.task_status
   - Condition: `"TaskA" in new_state.task_status`

2. ✅ Assertion: new_state.task_status["TaskA"].status == "running"
   - Condition: `new_state.task_status["TaskA"].status == "running"`

3. ✅ Assertion: new_state.task_status["TaskA"].started_ts == 12345.6
   - Condition: `new_state.task_status["TaskA"].started_ts == 12345.6`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 3/3 passed

#### tests/test_reducer_state_updates.py::test_reducer_log_line

- **Duration**: 0.003s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: len(new_state.logs) == 1
   - Condition: `len(new_state.logs) == 1`

2. ✅ Assertion: new_state.logs[0]["message"] == "hello"
   - Condition: `new_state.logs[0]["message"] == "hello"`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 2/2 passed

#### tests/test_task_smoke_outputs.py::test_t01_smoke

- **Duration**: 0.043s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: result.status == "ok"
   - Condition: `result.status == "ok"`

2. ✅ Assertion: os.path.exists(os.path.join(work_dir
   - Condition: `os.path.exists(os.path.join(work_dir`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 2/2 passed

#### tests/test_ui_non_gui_reducer_flow.py::test_ui_flow_sequence

- **Duration**: 0.003s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: state.is_running is True
   - Condition: `state.is_running is True`

2. ✅ Assertion: state.run_id == "run123"
   - Condition: `state.run_id == "run123"`

3. ✅ Assertion: state.task_status["T01"].status == "running"
   - Condition: `state.task_status["T01"].status == "running"`

4. ✅ Assertion: len(state.logs) == 1
   - Condition: `len(state.logs) == 1`

5. ✅ Assertion: state.task_status["T01"].status == "ok"
   - Condition: `state.task_status["T01"].status == "ok"`

6. ✅ Assertion: state.task_status["T01"].duration == 1.5
   - Condition: `state.task_status["T01"].duration == 1.5`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 6/6 passed

