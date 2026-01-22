import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Any
from src.core.ports.models import ModelPort

class CNNNet(nn.Module):
    def __init__(self, input_dim: int):
        super(CNNNet, self).__init__()
        # Simple 1D CNN for tabular data projection
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc = nn.Linear(32 * 8, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.softmax(x)

class TorchCNNModel(ModelPort):
    def __init__(self, feature_order: List[str], epochs: int = 10, batch_size: int = 64, lr: float = 0.001):
        self._feature_order = feature_order
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNNet(len(feature_order)).to(self.device)

    def train(self, X: Any, y: Any, **kwargs) -> None:
        X_t = torch.tensor(X.astype(np.float32)).to(self.device)
        y_t = torch.tensor(y.astype(np.int64)).to(self.device)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def predict_proba(self, X: Any) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(X.astype(np.float32)).to(self.device)
        with torch.no_grad():
            probs = self.model(X_t)
        return probs.cpu().numpy()

    def save(self, path: str) -> None:
        state = {
            "model_state": self.model.state_dict(),
            "feature_order": self._feature_order,
            "params": {"epochs": self.epochs, "batch_size": self.batch_size, "lr": self.lr}
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self._feature_order = state["feature_order"]
        self.epochs = state["params"]["epochs"]
        self.batch_size = state["params"]["batch_size"]
        self.lr = state["params"]["lr"]
        self.model = CNNNet(len(self._feature_order)).to(self.device)
        self.model.load_state_dict(state["model_state"])

    @property
    def feature_order(self) -> List[str]:
        return self._feature_order
