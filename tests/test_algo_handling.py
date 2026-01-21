"""
Tests for algorithm name handling utilities (get_algo_names, ensure_algo_column, sanitize_algo_name)
"""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.evaluation.visualizations import get_algo_names, ensure_algo_column, sanitize_algo_name


def test_get_algo_names_column():
    """
    Test get_algo_names() when 'algo' is a column.

    Input:
        - DataFrame with 'algo' column containing ['LR', 'DT', 'RF', 'CNN', 'TabNet']

    Processing:
        - Call get_algo_names() on DataFrame

    Expected Output:
        - Series with algorithm names as strings

    Method:
        - Direct call to get_algo_names()
    """
    df = pd.DataFrame({
        'algo': ['LR', 'DT', 'RF', 'CNN', 'TabNet'],
        'f1_mean': [0.9, 0.85, 0.92, 0.88, 0.87],
        'training_time_seconds': [1.2, 0.5, 2.3, 5.1, 4.2]
    })

    algos = get_algo_names(df)

    assert isinstance(algos, pd.Series), f"Expected Series, got {type(algos)}"
    assert len(algos) == 5, f"Expected 5 algorithms, got {len(algos)}"
    assert list(algos.values) == ['LR', 'DT', 'RF', 'CNN', 'TabNet'], (
        f"Expected ['LR', 'DT', 'RF', 'CNN', 'TabNet'], got {list(algos.values)}"
    )
    assert algos.dtype == 'object', f"Expected string dtype, got {algos.dtype}"


def test_get_algo_names_index():
    """
    Test get_algo_names() when 'algo' is index name.

    Input:
        - DataFrame with index named 'algo' containing ['LR', 'DT', 'RF']

    Processing:
        - Call get_algo_names() on DataFrame

    Expected Output:
        - Series with algorithm names as strings

    Method:
        - Direct call to get_algo_names()
    """
    df = pd.DataFrame({
        'f1_mean': [0.9, 0.85, 0.92],
        'training_time_seconds': [1.2, 0.5, 2.3]
    }, index=['LR', 'DT', 'RF'])
    df.index.name = 'algo'

    algos = get_algo_names(df)

    assert isinstance(algos, pd.Series), f"Expected Series, got {type(algos)}"
    assert len(algos) == 3, f"Expected 3 algorithms, got {len(algos)}"
    assert list(algos.values) == ['LR', 'DT', 'RF'], (
        f"Expected ['LR', 'DT', 'RF'], got {list(algos.values)}"
    )


def test_get_algo_names_raises():
    """
    Test get_algo_names() raises ValueError when algo not found.

    Input:
        - DataFrame without 'algo' column or index named 'algo'

    Processing:
        - Call get_algo_names() on DataFrame

    Expected Output:
        - ValueError raised

    Method:
        - Direct call to get_algo_names() with pytest.raises
    """
    df = pd.DataFrame({
        'f1_mean': [0.9, 0.85],
        'training_time_seconds': [1.2, 0.5]
    })

    with pytest.raises(ValueError, match="Algorithm names not found"):
        get_algo_names(df)


def test_ensure_algo_column():
    """
    Test ensure_algo_column() to ensure 'algo' is a column.

    Input:
        - DataFrame with 'algo' as column (should return unchanged)
        - DataFrame with 'algo' as index name (should reset_index)
        - None (should return None)

    Processing:
        - Call ensure_algo_column() on various inputs

    Expected Output:
        - DataFrame with 'algo' column, or None

    Method:
        - Direct call to ensure_algo_column()
    """
    # Test 1: DataFrame with 'algo' column (unchanged)
    df1 = pd.DataFrame({
        'algo': ['LR', 'DT'],
        'f1_mean': [0.9, 0.85]
    })
    result1 = ensure_algo_column(df1)
    assert result1 is not None
    assert 'algo' in result1.columns, "Expected 'algo' column"
    assert result1.equals(df1), "DataFrame with 'algo' column should be unchanged"

    # Test 2: DataFrame with 'algo' as index name (reset_index)
    df2 = pd.DataFrame({
        'f1_mean': [0.9, 0.85]
    }, index=['LR', 'DT'])
    df2.index.name = 'algo'
    result2 = ensure_algo_column(df2)
    assert result2 is not None
    assert 'algo' in result2.columns, "Expected 'algo' column after reset_index"
    assert list(result2['algo'].values) == ['LR', 'DT'], (
        f"Expected ['LR', 'DT'], got {list(result2['algo'].values)}"
    )

    # Test 3: None input (returns None)
    result3 = ensure_algo_column(None)
    assert result3 is None, "None input should return None"

    # Test 4: DataFrame without algo (raises ValueError)
    df4 = pd.DataFrame({'f1_mean': [0.9, 0.85]})
    with pytest.raises(ValueError, match="Algorithm names not found"):
        ensure_algo_column(df4)


def test_sanitize_algo_name():
    """
    Test sanitize_algo_name() to sanitize algorithm names for filenames.

    Input:
        - Various algorithm names with spaces, slashes, backslashes

    Processing:
        - Call sanitize_algo_name() on each input

    Expected Output:
        - Sanitized names with underscores replacing spaces/slashes

    Method:
        - Direct call to sanitize_algo_name()
    """
    test_cases = [
        ("LR", "LR"),
        ("Decision Tree", "Decision_Tree"),
        ("Random Forest", "Random_Forest"),
        ("CNN/TabNet", "CNN_TabNet"),
        ("CNN\\TabNet", "CNN_TabNet"),
        ("  CNN  ", "CNN"),
        ("Decision/Tree", "Decision_Tree"),
        ("Logistic Regression", "Logistic_Regression"),
    ]

    for input_name, expected in test_cases:
        result = sanitize_algo_name(input_name)
        assert result == expected, (
            f"For input '{input_name}', expected '{expected}', got '{result}'"
        )

    # Test official labels
    official_labels = ["LR", "DT", "RF", "CNN", "TabNet"]
    for label in official_labels:
        result = sanitize_algo_name(label)
        assert result == label, (
            f"Official label '{label}' should remain unchanged, got '{result}'"
        )
