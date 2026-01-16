#!/usr/bin/env python3
"""
CNN model for tabular data (DDoS detection)
Adapted for tabular network flow data according to IRP methodology
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from typing import Optional
import warnings

warnings.filterwarnings('ignore')


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular data"""
    
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Initialize tabular dataset
        
        Args:
            X: Feature array
            y: Target array (optional for inference)
        """
        self.X = torch.FloatTensor(X)
        if y is not None:
            self.y = torch.LongTensor(y)
        else:
            self.y = None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class TabularCNN(nn.Module):
    """
    Convolutional Neural Network adapted for tabular data
    Uses 1D convolutions on flattened feature vectors
    """
    
    def __init__(self, input_dim: int, num_classes: int = 2, hidden_dims: list = [64, 32, 16]):
        """
        Initialize CNN model
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes (default: 2 for binary classification)
            hidden_dims: List of hidden layer dimensions
        """
        super(TabularCNN, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Reshape input for 1D convolution (batch, channels, length)
        # We treat features as a sequence
        layers = []
        in_channels = 1
        
        # First conv layer
        layers.append(nn.Conv1d(in_channels, hidden_dims[0], kernel_size=3, padding=1))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(kernel_size=2))
        
        # Additional conv layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Conv1d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate flattened size after convolutions
        # Simplified calculation - may need adjustment based on input_dim
        flattened_size = hidden_dims[-1] * (input_dim // (2 ** len(hidden_dims)))
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Reshape for 1D convolution: (batch, 1, features)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        return x


class CNNTabularClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible CNN classifier for tabular data
    """
    
    def __init__(self, hidden_dims: list = [64, 32, 16], 
                 learning_rate: float = 0.001,
                 batch_size: int = 64,
                 epochs: int = 10,
                 device: Optional[str] = None,
                 random_state: int = 42):
        """
        Initialize CNN classifier
        
        Args:
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            device: Device to use ('cpu' or 'cuda'), None for auto-detect
            random_state: Random seed
        """
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.input_dim = None
    
    def _set_random_state(self):
        """Set random seeds for reproducibility"""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the CNN model
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            self
        """
        self._set_random_state()
        
        # Encode labels if needed
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.input_dim = X.shape[1]
        num_classes = len(np.unique(y_encoded))
        
        # Initialize model
        self.model = TabularCNN(
            input_dim=self.input_dim,
            num_classes=num_classes,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Create dataset and dataloader
        dataset = TabularDataset(X, y_encoded)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Test features
            
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        dataset = TabularDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch_X in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        
        # Decode labels
        predictions = np.array(predictions)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Test features
            
        Returns:
            Class probabilities
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        dataset = TabularDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        probabilities = []
        with torch.no_grad():
            for batch_X in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)


def main():
    """Test the CNN model"""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from dataset_loader import DatasetLoader
    from preprocessing_pipeline import PreprocessingPipeline
    
    loader = DatasetLoader()
    pipeline = PreprocessingPipeline(random_state=42)
    
    try:
        df = loader.load_ton_iot()
        X = df.drop(['label', 'type'] if 'type' in df.columns else ['label'], axis=1, errors='ignore')
        y = df['label']
        
        X_processed, y_processed = pipeline.prepare_data(X, y, apply_smote_flag=True, scale=True)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed
        )
        
        print(f"Training CNN on {X_train.shape[0]} samples...")
        
        # Train CNN
        cnn = CNNTabularClassifier(epochs=5, batch_size=64, random_state=42)
        cnn.fit(X_train, y_train)
        
        # Predict
        y_pred = cnn.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        
        print(f"\nCNN Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
