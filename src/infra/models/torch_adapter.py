import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from typing import Dict, Any
from src.core.ports.interfaces import IModelAdapter

class SimpleCNN(nn.Module):
    def __init__(self, input_dim: int):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        
        # Calculate linear input size
        self.fc1 = nn.Linear(64 * (input_dim // 2), 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1) # Add channel dim: [batch, 1, features]
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

class TorchAdapter(IModelAdapter):
    def train(self, X: pl.DataFrame, y: pl.Series, params: Dict[str, Any]) -> Any:
        input_dim = X.width
        model = SimpleCNN(input_dim)
        
        X_tensor = torch.FloatTensor(X.to_numpy())
        y_tensor = torch.FloatTensor(y.to_numpy()).view(-1, 1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=params.get("batch_size", 64), shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=params.get("lr", 0.001))
        criterion = nn.BCELoss()
        
        epochs = params.get("epochs", 5)
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        return model

    def predict_proba(self, model: Any, X: pl.DataFrame) -> pl.Series:
        model.eval()
        X_tensor = torch.FloatTensor(X.to_numpy())
        with torch.no_grad():
            probas = model(X_tensor).squeeze().numpy()
        return pl.Series("proba", probas)

    def save(self, model: Any, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)

    def load(self, path: str) -> Any:
        # Note: This requires knowing the input_dim to recreate the model
        # In a real scenario, we'd save metadata or use a factory
        # For now, we assume the model is already instantiated or we load state_dict into a provided model
        return torch.load(path)
