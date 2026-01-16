#!/usr/bin/env python3
"""
Data splitting utilities (stratified train/val/test split)
"""
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


def stratified_split(X, y, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> Tuple:
    """
    Perform stratified train/validation/test split
    
    Args:
        X: Features
        y: Labels
        test_size: Test set proportion
        val_size: Validation set proportion (relative to train+val)
        random_state: Random seed
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size_adjusted, 
        stratify=y_trainval, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
