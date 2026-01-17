import pytest
import numpy as np
import pandas as pd
from src.ahp_topsis_framework import AHPTopsisFramework, AHP, TOPSIS

def test_ahp_weights():
    criteria = ['C1', 'C2', 'C3']
    ahp = AHP(criteria)

    # C1 is 3x more important than C2, 2x more than C3
    # C2 is 0.5x as important as C3 (C3 is 2x more than C2)
    comparisons = {
        ('C1', 'C2'): 3.0,
        ('C1', 'C3'): 2.0,
        ('C2', 'C3'): 0.5
    }

    ahp.create_pairwise_matrix(comparisons)
    weights = ahp.compute_weights()

    assert len(weights) == 3
    assert np.isclose(np.sum(weights), 1.0)
    assert weights[0] > weights[2] > weights[1]
    assert ahp.is_consistent()

def test_topsis_ranking():
    # 3 alternatives, 2 criteria (both max)
    decision_matrix = np.array([
        [0.9, 0.5],
        [0.8, 0.8],
        [0.5, 0.9]
    ])
    weights = np.array([0.5, 0.5])

    topsis = TOPSIS(decision_matrix, weights)
    scores, rankings = topsis.rank()

    assert len(scores) == 3
    assert len(rankings) == 3
    # Alternative 1 (index 1, [0.8, 0.8]) should be best as it's balanced
    assert rankings[0] == 1

def test_ahp_topsis_framework():
    criteria = ['Perf', 'Res', 'Exp']
    framework = AHPTopsisFramework(criteria)

    comparisons = {
        ('Perf', 'Res'): 3.0,
        ('Perf', 'Exp'): 5.0,
        ('Res', 'Exp'): 2.0
    }
    framework.set_ahp_comparisons(comparisons)

    decision_matrix = np.array([
        [0.99, 0.2, 0.1], # High perf, high resource use, low exp
        [0.80, 0.8, 0.9], # Good balance
    ])
    framework.set_decision_matrix(decision_matrix, ['Model_A', 'Model_B'])

    results = framework.rank_alternatives()

    assert 'Alternative' in results.columns
    assert 'Score' in results.columns
    assert 'Rank' in results.columns
    assert len(results) == 2
