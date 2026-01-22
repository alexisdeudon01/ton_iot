from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np

class ModelPort(ABC):
    @abstractmethod
    def train(self, X: Any, y: Any, **kwargs) -> None:
        pass

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @property
    @abstractmethod
    def feature_order(self) -> List[str]:
        pass
