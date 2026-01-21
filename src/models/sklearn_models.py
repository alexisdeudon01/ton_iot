#!/usr/bin/env python3
"""
Sklearn models: Logistic Regression, Decision Tree, Random Forest
"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from typing import Optional


def make_lr(random_state: int = 42, max_iter: int = 1000, class_weight: Optional[str] = None) -> LogisticRegression:
    """
    Create Logistic Regression model

    Args:
        random_state: Random seed
        max_iter: Maximum iterations
        class_weight: Class weights ('balanced' or None)
    """
    params = {
        'max_iter': max_iter,
        'random_state': random_state,
        'n_jobs': -1,
        'solver': 'lbfgs'  # Explicitly set solver to avoid warnings and ensure compatibility
    }
    if class_weight:
        params['class_weight'] = class_weight

    return LogisticRegression(**params)


def make_dt(random_state: int = 42, max_depth: Optional[int] = None,
            min_samples_leaf: int = 1) -> DecisionTreeClassifier:
    """
    Create Decision Tree model

    Args:
        random_state: Random seed
        max_depth: Maximum tree depth (None = unlimited)
        min_samples_leaf: Minimum samples per leaf
    """
    return DecisionTreeClassifier(
        random_state=random_state,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf
    )


def make_rf(random_state: int = 42, n_estimators: int = 100) -> RandomForestClassifier:
    """
    Create Random Forest model

    Args:
        random_state: Random seed
        n_estimators: Number of trees
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )


def make_knn(n_neighbors: int = 5, weights: str = 'uniform') -> KNeighborsClassifier:
    """
    Create KNN model

    Args:
        n_neighbors: Number of neighbors
        weights: Weight function
    """
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        n_jobs=-1
    )
