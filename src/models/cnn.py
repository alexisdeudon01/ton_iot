#!/usr/bin/env python3
"""
CNN model for tabular data (DDoS detection)
Adapted for tabular network flow data according to IRP methodology.

CNN is mandatory: requires torch.
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING, Any, List, Optional, Type, Union, cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

TORCH_AVAILABLE = True

# Type-safe base classes for Pylance
if TYPE_CHECKING:
    import torch.nn as nn_types
    from torch.utils.data import Dataset as TorchDataset

    _BaseDataset = TorchDataset
    _BaseModule = nn_types.Module
else:
    _BaseDataset = Dataset
    _BaseModule = nn.Module


class TabularDataset(cast(Type, _BaseDataset)):
    """PyTorch Dataset for tabular data."""

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Initialize tabular dataset.

        Args:
            X: Feature array
            y: Target array (optional for inference)
        """
        self.X = torch.as_tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.as_tensor(y, dtype=torch.long)
        else:
            self.y = None

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class TabularCNN(cast(Type, _BaseModule)):
    """
    CNN adapted for tabular data:
    Treat features as 1D sequence, apply Conv1d over the feature dimension.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dims: Optional[List[int]] = None,
    ):
        """
        Args:
            input_dim: Number of input features
            num_classes: Number of classes
            hidden_dims: Conv channel sizes
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        # Validate hidden_dims is not empty
        if not hidden_dims or len(hidden_dims) == 0:
            raise ValueError(
                "hidden_dims cannot be empty. Provide at least one layer size "
                "(e.g., [64]) or use None for default [64, 32, 16]"
            )

        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)

        layers = []
        in_channels = 1
        for i, out_ch in enumerate(hidden_dims):
            layers.append(nn.Conv1d(in_channels, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = out_ch

        self.conv_layers = nn.Sequential(*layers)

        # Compute flattened size after pooling
        # After each MaxPool1d(2), length is halved (floor)
        pooled_len = self.input_dim
        for _ in hidden_dims:
            pooled_len = max(1, pooled_len // 2)

        # Use last hidden dim for flattened size (safe since we validated non-empty)
        flattened_size = hidden_dims[-1] * pooled_len

        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            logits: (batch_size, num_classes)
        """
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.conv_layers(x)  # (batch, channels, length)
        x = x.reshape(x.size(0), -1)
        return self.fc_layers(x)


class CNNTabularClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible CNN classifier for tabular data.
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 10,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.random_state = random_state

        self.model: TabularCNN = cast(TabularCNN, None)
        self.label_encoder = LabelEncoder()
        self.input_dim: Optional[int] = None
        self.device_obj: Optional[torch.device] = None
        
        logger.info(f"CNNTabularClassifier initialized: hidden_dims={hidden_dims}, lr={learning_rate}, "
                   f"batch_size={batch_size}, epochs={epochs}, device={device}, random_state={random_state}")

    def _set_random_state(self) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNNTabularClassifier":
        logger.info(f"Starting CNN training: X.shape={X.shape}, y.shape={y.shape}, unique_classes={len(np.unique(y))}")
        self._set_random_state()

        if self.device is None:
            self.device_obj = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device_obj = torch.device(self.device)
        
        logger.info(f"Using device: {self.device_obj}")

        y_encoded = self.label_encoder.fit_transform(y)
        self.input_dim = int(X.shape[1])
        num_classes = int(len(np.unique(y_encoded)))
        
        logger.debug(f"Encoded labels: {len(np.unique(y_encoded))} classes, input_dim={self.input_dim}")

        hidden_dims = (
            self.hidden_dims if self.hidden_dims is not None else [64, 32, 16]
        )
        
        logger.debug(f"Creating TabularCNN with hidden_dims={hidden_dims}, num_classes={num_classes}")

        self.model = TabularCNN(
            input_dim=self.input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
        ).to(self.device_obj)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        dataset = TabularDataset(X, cast(np.ndarray, y_encoded))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        logger.info(f"DataLoader created: batch_size={self.batch_size}, batches={len(dataloader)}")

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device_obj)
                batch_y = batch_y.to(self.device_obj)

                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            if (epoch + 1) % 5 == 0 or (epoch + 1) == self.epochs:
                logger.info(
                    f"[CNN] Epoch {epoch+1}/{self.epochs} - avg_loss={avg_loss:.4f}, batches={n_batches}"
                )
            else:
                logger.debug(f"[CNN] Epoch {epoch+1}/{self.epochs} - avg_loss={avg_loss:.4f}")
        
        logger.info(f"CNN training completed after {self.epochs} epochs")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.device_obj is None:
            raise ValueError("Model must be fitted before prediction")

        logger.debug(f"CNN predict: X.shape={X.shape}, batch_size={self.batch_size}")
        self.model.eval()
        dataset = TabularDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        preds = []
        with torch.no_grad():
            for batch_X in dataloader:
                batch_X = batch_X.to(self.device_obj)
                logits = self.model(batch_X)
                pred = torch.argmax(logits, dim=1)
                preds.extend(pred.cpu().numpy())

        preds = np.asarray(preds)
        result = self.label_encoder.inverse_transform(preds)
        logger.debug(f"CNN prediction completed: {len(result)} predictions, unique_classes={len(np.unique(result))}")
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.device_obj is None:
            raise ValueError("Model must be fitted before prediction")

        logger.debug(f"CNN predict_proba: X.shape={X.shape}, batch_size={self.batch_size}")
        self.model.eval()
        dataset = TabularDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        probs_all = []
        with torch.no_grad():
            for batch_X in dataloader:
                batch_X = batch_X.to(self.device_obj)
                logits = self.model(batch_X)
                probs = torch.softmax(logits, dim=1)
                probs_all.extend(probs.cpu().numpy())

        result = np.asarray(probs_all)
        logger.debug(f"CNN predict_proba completed: shape={result.shape}, probs_sum_range=[{result.sum(axis=1).min():.3f}, {result.sum(axis=1).max():.3f}]")
        return result
