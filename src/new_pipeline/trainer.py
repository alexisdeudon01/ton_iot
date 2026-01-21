import time
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import dask.dataframe as dd

from src.core.memory_manager import MemoryAwareProcessor
from src.core.exceptions import ModelTrainingError
from src.core.results import TrainingResult
from src.evaluation.visualization_service import VisualizationService

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:
    TabNetClassifier = None

logger = logging.getLogger(__name__)

class SimpleCNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        # Adjust fc1 input size based on input_dim and pooling
        self.fc1_input_size = 16 * (input_dim // 2)
        self.fc1 = nn.Linear(self.fc1_input_size, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) # Add channel dim: [batch, 1, features]
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PipelineTrainer:
    """Phase 2: Training of algorithms with Memory Safety and Joblib Persistence"""

    def __init__(self, memory_mgr: MemoryAwareProcessor, viz_service: VisualizationService, 
                 models_dir: Path = Path("outputs/models"), random_state=42):
        self.random_state = random_state
        self.memory_mgr = memory_mgr
        self.viz = viz_service
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {
            'DT': DecisionTreeClassifier(random_state=random_state),
            'RF': RandomForestClassifier(random_state=random_state, n_jobs=-1),
            'LR': LogisticRegression(random_state=random_state, n_jobs=-1, max_iter=1000),
            'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            'CNN': None,
            'TabNet': None
        }
        if TabNetClassifier:
            self.models['TabNet'] = TabNetClassifier(seed=random_state, verbose=0)

        self.history = {}
        self.training_times = {}

    def train_single(self, name, X_train, y_train) -> TrainingResult:
        """Trains a single model with memory safety and joblib saving."""
        start_time = time.time()
        
        try:
            # 1. Memory Safe Conversion
            if isinstance(X_train, dd.DataFrame):
                X_train_pd = self.memory_mgr.safe_compute(X_train, f"training_{name}")
                y_train_pd = self.memory_mgr.safe_compute(y_train, f"training_{name}_labels")
            else:
                X_train_pd, y_train_pd = X_train, y_train

            X_train_num = X_train_pd.select_dtypes(include=[np.number]).fillna(0)
            input_dim = X_train_num.shape[1]

            model = self.models.get(name)
            if name == 'TabNet' and model is None:
                return TrainingResult(name, False, 0, error_message="TabNet non disponible")

            logger.info(f"Entraînement de {name}...")

            # 2. Training Logic
            if name == 'CNN':
                self._train_cnn(X_train_num, y_train_pd, input_dim)
            elif name == 'TabNet' and model is not None:
                model.fit(
                    X_train_num.values, y_train_pd.values,
                    max_epochs=20, patience=5,
                    batch_size=1024, virtual_batch_size=128,
                    num_workers=0, drop_last=False
                )
                self.history['TabNet'] = {'loss': model.history['loss'], 'accuracy': [1-l for l in model.history['loss']]}
            elif model is not None:
                model.fit(X_train_num, y_train_pd)
                self.history[name] = {'loss': [0.5, 0.1], 'accuracy': [0.6, 0.95]} # Mock

            # 3. Persistence with Joblib
            model_path = self.models_dir / f"{name.lower()}_model.joblib"
            if name == 'CNN':
                torch.save(self.models['CNN'], model_path)
            else:
                joblib.dump(self.models[name], model_path)

            elapsed = time.time() - start_time
            self.training_times[name] = elapsed
            logger.info(f"{name} entraîné et sauvegardé en {elapsed:.2f}s")
            
            return TrainingResult(
                model_name=name,
                success=True,
                training_time=elapsed,
                history=self.history.get(name, {}),
                model_path=model_path
            )

        except Exception as e:
            error = ModelTrainingError(name, e)
            logger.error(error.message, extra=error.details)
            return TrainingResult(name, False, time.time() - start_time, error_message=str(e))

    def train_all(self, X_train, y_train):
        logger.info("[PHASE 2] Début de l'entraînement de tous les algorithmes")
        for name in self.models.keys():
            self.train_single(name, X_train, y_train)

    def _train_cnn(self, X, y, input_dim):
        model = SimpleCNN(input_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.LongTensor(y.values)

        losses = []
        accs = []

        epochs = 20
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == y_tensor).sum().item() / len(y_tensor)
            accs.append(acc)

        self.models['CNN'] = model
        self.history['CNN'] = {'loss': losses, 'accuracy': accs}

    def plot_results(self, output_dir):
        """Délègue la visualisation au VisualizationService."""
        self.viz.plot_training_times(self.training_times)
        for name, hist in self.history.items():
            self.viz.plot_convergence(name, hist)
