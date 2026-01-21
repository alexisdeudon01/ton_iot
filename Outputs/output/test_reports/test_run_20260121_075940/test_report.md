# Test Execution Report

**Generated**: 2026-01-21 08:01:35

## Summary Statistics

- **Total Tests**: 5
- **Passed**: 5 (100.0%)
- **Failed**: 0 (0.0%)
- **Skipped**: 0 (0.0%)

## Test Results by Outcome

### ✅ Passed Tests

#### tests/test_algo_handling.py::test_get_algo_names_column

- **Duration**: 0.005s
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

- **Duration**: 0.004s
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

- **Duration**: 0.004s
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

- **Duration**: 0.002s
- **Input**: N/A
**Validation Criteria (All Passed):**

1. ✅ Assertion: result == expected
   - Condition: `result == expected`

2. ✅ Assertion: result == label
   - Condition: `result == label`

- **Expected Output**: N/A
- **Success Reason**: All validation criteria satisfied: 2/2 passed

