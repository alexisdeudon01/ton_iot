from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LogWinsorizer(BaseEstimator, TransformerMixin):
    """Clip extremes per feature then apply log1p.

    Designed for non-negative heavy-tailed features.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99, clip_min: float = 0.0):
        self.lower = lower
        self.upper = upper
        self.clip_min = clip_min
        self.lower_bounds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X_np = self._to_numpy(X)
        self.lower_bounds_ = np.nanquantile(X_np, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(X_np, self.upper, axis=0)

        if self.clip_min is not None:
            self.lower_bounds_ = np.maximum(self.lower_bounds_, self.clip_min)
        self.upper_bounds_ = np.maximum(self.upper_bounds_, self.lower_bounds_)
        return self

    def transform(self, X):
        X_np = self._to_numpy(X)
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise ValueError("LogWinsorizer must be fitted before calling transform().")
        X_clipped = np.clip(X_np, self.lower_bounds_, self.upper_bounds_)
        if self.clip_min is not None:
            X_clipped = np.maximum(X_clipped, self.clip_min)
        return np.log1p(X_clipped)

    @staticmethod
    def _to_numpy(X):
        if hasattr(X, "to_numpy"):
            return X.to_numpy()
        return np.asarray(X)
