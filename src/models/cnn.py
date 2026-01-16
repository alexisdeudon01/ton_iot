#!/usr/bin/env python3
"""
CNN model for tabular data (optional - requires torch)
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Try to import torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch not available - CNN will be skipped. Install via: pip install -r requirements-nn.txt")


if TORCH_AVAILABLE:
    class TabularDataset(Dataset):
        """PyTorch Dataset for tabular data"""
        def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y) if y is not None else None
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]
    
    
    class TabularCNN(nn.Module):
        """CNN for tabular data"""
        def __init__(self, input_dim: int, num_classes: int = 2, hidden_dims: list = [64, 32, 16]):
            super().__init__()
            self.input_dim = input_dim
            layers = []
            in_channels = 1
            for hdim in hidden_dims:
                layers.extend([
                    nn.Conv1d(in_channels, hdim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hdim),
                    nn.ReLU(),
                    nn.MaxPool1d(2)
                ])
                in_channels = hdim
            self.conv = nn.Sequential(*layers)
            # Calculate flattened size
            with torch.no_grad():
                dummy = torch.zeros(1, 1, input_dim)
                out = self.conv(dummy)
                flattened_size = out.view(1, -1).shape[1]
            self.fc = nn.Sequential(
                nn.Linear(flattened_size, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, num_classes)
            )
        
        def forward(self, x):
            x = x.unsqueeze(1)
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    
    class CNNTabularClassifier(BaseEstimator, ClassifierMixin):
        """Sklearn-compatible CNN classifier"""
        def __init__(self, hidden_dims=[64, 32, 16], learning_rate=0.001,
                     batch_size=64, epochs=10, device=None, random_state=42):
            if not TORCH_AVAILABLE:
                raise ImportError("CNNTabularClassifier requires torch. Install via: pip install -r requirements-nn.txt")
            self.hidden_dims = hidden_dims
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.epochs = epochs
            self.random_state = random_state
            self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = None
            self.label_encoder = LabelEncoder()
        
        def _set_random_state(self):
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_state)
        
        def fit(self, X, y):
            self._set_random_state()
            y_enc = self.label_encoder.fit_transform(y)
            self.model = TabularCNN(X.shape[1], len(np.unique(y_enc)), self.hidden_dims).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            dataset = TabularDataset(X, y_enc)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            self.model.train()
            for epoch in range(self.epochs):
                for bx, by in loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(self.model(bx), by)
                    loss.backward()
                    optimizer.step()
            return self
        
        def predict(self, X):
            if self.model is None:
                raise ValueError("Model not fitted")
            self.model.eval()
            dataset = TabularDataset(X)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            preds = []
            with torch.no_grad():
                for bx in loader:
                    bx = bx.to(self.device)
                    _, p = torch.max(self.model(bx), 1)
                    preds.extend(p.cpu().numpy())
            return self.label_encoder.inverse_transform(np.array(preds))
        
        def predict_proba(self, X):
            if self.model is None:
                raise ValueError("Model not fitted")
            self.model.eval()
            dataset = TabularDataset(X)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            probs = []
            with torch.no_grad():
                for bx in loader:
                    bx = bx.to(self.device)
                    p = torch.softmax(self.model(bx), dim=1)
                    probs.extend(p.cpu().numpy())
            return np.array(probs)

else:
    class CNNTabularClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, *args, **kwargs):
            raise ImportError("CNNTabularClassifier requires torch. Install via: pip install -r requirements-nn.txt")
