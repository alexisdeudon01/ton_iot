#!/usr/bin/env python3
"""
AHP-TOPSIS Multi-Criteria Decision Making Framework
Implements Analytic Hierarchy Process (AHP) for criteria weighting
and Technique for Order Preference by Similarity to Ideal Solution (TOPSIS) for ranking
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class AHP:
    """Analytic Hierarchy Process for criteria weighting"""

    def __init__(self, criteria_names: List[str]):
        """
        Initialize AHP

        Args:
            criteria_names: List of criterion names
        """
        self.criteria_names = criteria_names
        self.n_criteria = len(criteria_names)
        self.pairwise_matrix = None
        self.weights = None
        self.consistency_ratio = None

    def create_pairwise_matrix(self, comparisons: Dict[Tuple[str, str], float]) -> np.ndarray:
        """
        Create pairwise comparison matrix from comparisons dictionary

        Args:
            comparisons: Dictionary mapping (criterion1, criterion2) to comparison value
                        Comparison values follow Saaty scale (1-9):
                        1 = equal importance
                        3 = moderate importance
                        5 = strong importance
                        7 = very strong importance
                        9 = extreme importance
                        Or reciprocals for inverse comparisons

        Returns:
            Pairwise comparison matrix
        """
        matrix = np.ones((self.n_criteria, self.n_criteria))

        # Create index mapping
        idx_map = {name: idx for idx, name in enumerate(self.criteria_names)}

        # Fill matrix from comparisons
        for (c1, c2), value in comparisons.items():
            i = idx_map[c1]
            j = idx_map[c2]
            matrix[i, j] = value
            matrix[j, i] = 1.0 / value

        self.pairwise_matrix = matrix
        return matrix

    def compute_weights(self, pairwise_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute criteria weights using eigenvalue method

        Args:
            pairwise_matrix: Pairwise comparison matrix (uses self.pairwise_matrix if None)

        Returns:
            Normalized weight vector
        """
        if pairwise_matrix is None:
            if self.pairwise_matrix is None:
                raise ValueError("No pairwise matrix provided")
            pairwise_matrix = self.pairwise_matrix

        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(pairwise_matrix)

        # Get real part (in case of complex eigenvalues)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        # Find maximum eigenvalue and corresponding eigenvector
        max_idx = np.argmax(eigenvalues)
        max_eigenvalue = eigenvalues[max_idx]
        principal_eigenvector = eigenvectors[:, max_idx]

        # Normalize eigenvector to get weights
        weights = principal_eigenvector / np.sum(principal_eigenvector)
        weights = np.real(weights)  # Ensure real values

        self.weights = weights

        # Compute consistency ratio
        self.consistency_ratio = self._compute_consistency_ratio(pairwise_matrix, max_eigenvalue)

        return weights

    def _compute_consistency_ratio(self, pairwise_matrix: np.ndarray,
                                   max_eigenvalue: float) -> float:
        """
        Compute consistency ratio (CR) for AHP

        Args:
            pairwise_matrix: Pairwise comparison matrix
            max_eigenvalue: Maximum eigenvalue

        Returns:
            Consistency ratio
        """
        n = self.n_criteria
        ci = (max_eigenvalue - n) / (n - 1)  # Consistency Index

        # Random Index (RI) values for different matrix sizes (Saaty, 1980)
        ri_values = {
            1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
        }
        ri = ri_values.get(n, 1.49)  # Default for n > 10

        cr = ci / ri if ri > 0 else 0.0  # Consistency Ratio
        return cr

    def is_consistent(self, threshold: float = 0.1) -> bool:
        """
        Check if pairwise matrix is consistent

        Args:
            threshold: Maximum acceptable CR (default: 0.1)

        Returns:
            True if consistent, False otherwise
        """
        if self.consistency_ratio is None:
            if self.weights is None:
                self.compute_weights()
        return self.consistency_ratio <= threshold


class TOPSIS:
    """Technique for Order Preference by Similarity to Ideal Solution"""

    def __init__(self, decision_matrix: np.ndarray, weights: np.ndarray,
                 criteria_types: List[str] = None):
        """
        Initialize TOPSIS

        Args:
            decision_matrix: Decision matrix (alternatives x criteria)
                           Each row is an alternative, each column is a criterion
            weights: Criteria weights (must sum to 1)
            criteria_types: List indicating 'max' or 'min' for each criterion
                          If None, assumes all are 'max'
        """
        self.decision_matrix = decision_matrix.copy()
        self.n_alternatives, self.n_criteria = decision_matrix.shape
        self.weights = weights / np.sum(weights)  # Normalize weights

        if criteria_types is None:
            self.criteria_types = ['max'] * self.n_criteria
        else:
            self.criteria_types = criteria_types

        if len(self.criteria_types) != self.n_criteria:
            raise ValueError(f"criteria_types length ({len(self.criteria_types)}) "
                           f"must match number of criteria ({self.n_criteria})")

    def normalize_matrix(self) -> np.ndarray:
        """
        Normalize decision matrix using vector normalization

        Returns:
            Normalized decision matrix
        """
        normalized = np.zeros_like(self.decision_matrix, dtype=float)

        for j in range(self.n_criteria):
            column = self.decision_matrix[:, j]
            norm = np.sqrt(np.sum(column ** 2))
            if norm > 0:
                normalized[:, j] = column / norm
            else:
                normalized[:, j] = 0

        return normalized

    def find_ideal_solutions(self, normalized_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find positive ideal solution (PIS) and negative ideal solution (NIS)

        Args:
            normalized_matrix: Normalized decision matrix

        Returns:
            Tuple of (positive_ideal_solution, negative_ideal_solution)
        """
        weighted_matrix = normalized_matrix * self.weights

        positive_ideal = np.zeros(self.n_criteria)
        negative_ideal = np.zeros(self.n_criteria)

        for j in range(self.n_criteria):
            if self.criteria_types[j] == 'max':
                positive_ideal[j] = np.max(weighted_matrix[:, j])
                negative_ideal[j] = np.min(weighted_matrix[:, j])
            else:  # 'min'
                positive_ideal[j] = np.min(weighted_matrix[:, j])
                negative_ideal[j] = np.max(weighted_matrix[:, j])

        return positive_ideal, negative_ideal

    def compute_distances(self, normalized_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute distances to positive and negative ideal solutions

        Args:
            normalized_matrix: Normalized decision matrix

        Returns:
            Tuple of (distances_to_positive, distances_to_negative)
        """
        weighted_matrix = normalized_matrix * self.weights
        positive_ideal, negative_ideal = self.find_ideal_solutions(normalized_matrix)

        distances_positive = np.sqrt(np.sum((weighted_matrix - positive_ideal) ** 2, axis=1))
        distances_negative = np.sqrt(np.sum((weighted_matrix - negative_ideal) ** 2, axis=1))

        return distances_positive, distances_negative

    def compute_scores(self, distances_positive: np.ndarray,
                      distances_negative: np.ndarray) -> np.ndarray:
        """
        Compute TOPSIS scores (closeness coefficient)

        Args:
            distances_positive: Distances to positive ideal solution
            distances_negative: Distances to negative ideal solution

        Returns:
            TOPSIS scores (higher is better)
        """
        total_distance = distances_positive + distances_negative
        scores = distances_negative / (total_distance + 1e-10)  # Add small epsilon to avoid division by zero
        return scores

    def rank(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform TOPSIS ranking

        Returns:
            Tuple of (scores, rankings)
            Rankings are indices sorted by score (best first)
        """
        normalized_matrix = self.normalize_matrix()
        distances_positive, distances_negative = self.compute_distances(normalized_matrix)
        scores = self.compute_scores(distances_positive, distances_negative)

        # Rankings (descending order: highest score = rank 1)
        rankings = np.argsort(scores)[::-1]

        return scores, rankings


class AHPTopsisFramework:
    """Complete AHP-TOPSIS framework for multi-criteria decision making"""

    def __init__(self, criteria_names: List[str], criteria_types: List[str] = None):
        """
        Initialize AHP-TOPSIS framework

        Args:
            criteria_names: List of criterion names
            criteria_types: List of 'max' or 'min' for each criterion (default: all 'max')
        """
        self.criteria_names = criteria_names
        self.criteria_types = criteria_types or ['max'] * len(criteria_names)
        self.ahp = AHP(criteria_names)
        self.weights = None
        self.decision_matrix = None
        self.rankings = None
        self.scores = None

    def set_ahp_comparisons(self, comparisons: Dict[Tuple[str, str], float]):
        """
        Set AHP pairwise comparisons and compute weights

        Args:
            comparisons: Dictionary mapping (criterion1, criterion2) to comparison value
        """
        self.ahp.create_pairwise_matrix(comparisons)
        self.weights = self.ahp.compute_weights()

        # Check consistency
        if not self.ahp.is_consistent():
            print(f"Warning: AHP consistency ratio ({self.ahp.consistency_ratio:.4f}) > 0.1")
            print("Consider revising pairwise comparisons")
        else:
            print(f"AHP consistency ratio: {self.ahp.consistency_ratio:.4f} (OK)")

    def set_decision_matrix(self, decision_matrix: np.ndarray, alternative_names: Optional[List[str]] = None):
        """
        Set decision matrix (alternatives x criteria)

        Args:
            decision_matrix: Decision matrix
            alternative_names: Names of alternatives (rows)
        """
        self.decision_matrix = decision_matrix
        self.alternative_names = alternative_names if alternative_names is not None else [f"Alternative_{i+1}" for i in range(len(decision_matrix))]

    def rank_alternatives(self) -> pd.DataFrame:
        """
        Perform AHP-TOPSIS ranking

        Returns:
            DataFrame with rankings and scores
        """
        if self.weights is None:
            raise ValueError("AHP weights must be computed first (call set_ahp_comparisons)")

        if self.decision_matrix is None:
            raise ValueError("Decision matrix must be set first (call set_decision_matrix)")

        # Run TOPSIS
        topsis = TOPSIS(self.decision_matrix, self.weights, self.criteria_types)
        scores, rankings = topsis.rank()

        self.scores = scores
        self.rankings = rankings

        # Create results DataFrame
        results = pd.DataFrame({
            'Alternative': [self.alternative_names[i] for i in rankings],
            'Score': scores[rankings],
            'Rank': range(1, len(rankings) + 1)
        })

        return results

    def get_weights(self) -> pd.DataFrame:
        """Get AHP weights as DataFrame"""
        return pd.DataFrame({
            'Criterion': self.criteria_names,
            'Weight': self.weights
        })


def main():
    """Test the AHP-TOPSIS framework"""
    # Example: Ranking algorithms based on 3 criteria
    criteria = ['Detection Performance', 'Resource Efficiency', 'Explainability']

    # Initialize framework
    framework = AHPTopsisFramework(
        criteria_names=criteria,
        criteria_types=['max', 'max', 'max']  # All are "higher is better"
    )

    # Set AHP comparisons (example: Detection Performance is moderately more important than Resource Efficiency)
    comparisons = {
        ('Detection Performance', 'Resource Efficiency'): 3,  # Detection Performance is moderately more important
        ('Detection Performance', 'Explainability'): 2,  # Detection Performance is slightly more important
        ('Resource Efficiency', 'Explainability'): 1/2  # Explainability is slightly more important than Resource Efficiency
    }
    framework.set_ahp_comparisons(comparisons)

    # Display weights
    print("\nAHP Weights:")
    print(framework.get_weights())

    # Example decision matrix (5 algorithms x 3 criteria)
    # Values should be normalized to similar scales
    decision_matrix = np.array([
        [0.95, 0.80, 0.90],  # Algorithm 1
        [0.90, 0.90, 0.85],  # Algorithm 2
        [0.85, 0.70, 0.95],  # Algorithm 3
        [0.88, 0.85, 0.75],  # Algorithm 4
        [0.92, 0.75, 0.80],  # Algorithm 5
    ])

    alternative_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'CNN', 'TabNet']

    framework.set_decision_matrix(decision_matrix, alternative_names)

    # Rank alternatives
    results = framework.rank_alternatives()

    print("\nAHP-TOPSIS Ranking Results:")
    print(results)


if __name__ == "__main__":
    main()
