#!/usr/bin/env python3
"""
TabNet model for tabular data (DDoS detection)
Uses pytorch-tabnet library for implementation
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from typing import Optional
import warnings
import torch

warnings.filterwarnings('ignore')

# Try to import pytorch-tabnet, with fallback if not available
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("Warning: pytorch-tabnet not installed. TabNet model will not be available.")
    print("Install with: pip install pytorch-tabnet")


class TabNetClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible TabNet classifier wrapper
    """

    def __init__(self, n_d: int = 8, n_a: int = 8,
                 n_steps: int = 3,
                 gamma: float = 1.5,
                 lambda_sparse: float = 1e-3,
                 optimizer_fn: Optional[object] = None,
                 optimizer_params: Optional[dict] = None,
                 scheduler_fn: Optional[object] = None,
                 scheduler_params: Optional[dict] = None,
                 mask_type: str = 'sparsemax',
                 n_shared: int = 2,
                 n_independent: int = 2,
                 virtual_batch_size: int = 128,
                 momentum: float = 0.02,
                 clip_value: float = 2.0,
                 verbose: int = 0,
                 seed: int = 42,
                 max_epochs: int = 100,
                 patience: int = 15,
                 batch_size: int = 1024):
        """
        Initialize TabNet classifier

        Args:
            n_d: Dimension of the decision layer
            n_a: Dimension of the attention layer
            n_steps: Number of steps in the encoder
            gamma: Coefficient for feature reusage in the masks
            lambda_sparse: Coefficient for sparsity regularization
            optimizer_fn: Optimizer function (default: torch.optim.Adam)
            optimizer_params: Optimizer parameters
            scheduler_fn: Learning rate scheduler (optional)
            scheduler_params: Scheduler parameters
            mask_type: Type of mask ('sparsemax' or 'entmax')
            n_shared: Number of shared layers in the feature transformer
            n_independent: Number of independent layers in the feature transformer
            virtual_batch_size: Size of the virtual batch for ghost batch norm
            momentum: Momentum for batch normalization
            clip_value: Value to clip gradients
            verbose: Verbosity level
            seed: Random seed
            max_epochs: Maximum number of training epochs
            patience: Patience for early stopping
            batch_size: Batch size for training
        """
        if not TABNET_AVAILABLE:
            raise ImportError(
                "pytorch-tabnet is not installed. Install with: pip install pytorch-tabnet"
            )

        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.optimizer_fn = optimizer_fn or torch.optim.Adam if TABNET_AVAILABLE else None
        self.optimizer_params = optimizer_params or {'lr': 2e-2}
        self.scheduler_fn = scheduler_fn
        self.scheduler_params = scheduler_params
        self.mask_type = mask_type
        self.n_shared = n_shared
        self.n_independent = n_independent
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.clip_value = clip_value
        self.verbose = verbose
        self.seed = seed
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size

        self.model = None
        self.label_encoder = LabelEncoder()
        self.input_dim = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the TabNet model

        Args:
            X: Training features
            y: Training labels

        Returns:
            self
        """
        # Encode labels if needed
        y_encoded = self.label_encoder.fit_transform(y)
        self.input_dim = X.shape[1]

        # Initialize TabNet model
        self.model = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            lambda_sparse=self.lambda_sparse,
            optimizer_fn=self.optimizer_fn,
            optimizer_params=self.optimizer_params,
            scheduler_fn=self.scheduler_fn,
            scheduler_params=self.scheduler_params,
            mask_type=self.mask_type,
            n_shared=self.n_shared,
            n_independent=self.n_independent,
            momentum=self.momentum,
            clip_value=self.clip_value,
            seed=self.seed,
            verbose=self.verbose
        )

        # Train model
        self.model.fit(
            X_train=X,
            y_train=y_encoded,
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size
        )

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

        predictions = self.model.predict(X)
        predictions = predictions.flatten()

        # Decode labels
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

        probabilities = self.model.predict_proba(X)
        return probabilities


# Fallback implementation if TabNet is not available
if not TABNET_AVAILABLE:
    class TabNetClassifierWrapper(BaseEstimator, ClassifierMixin):
        """Fallback TabNet classifier that raises error on use"""

        def __init__(self, **kwargs):
            raise ImportError(
                "pytorch-tabnet is not installed. Install with: pip install pytorch-tabnet"
            )


def main():
    """Test the TabNet model"""
    if not TABNET_AVAILABLE:
        print("TabNet is not available. Install pytorch-tabnet first.")
        return

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from dataset_loader import DatasetLoader
    from preprocessing_pipeline import PreprocessingPipeline

    loader = DatasetLoader()
    pipeline = PreprocessingPipeline(random_state=42)

    try:
        df = loader.load_ton_iot(sample_ratio=0.01)
        X = df.drop(['label', 'type'] if 'type' in df.columns else ['label'], axis=1, errors='ignore')
        y = df['label']

        result = pipeline.prepare_data(
            X, y,
            apply_resampling=True,
            apply_scaling=True,
            apply_splitting=False
        )
        X_processed = result['X_processed']
        y_processed = result['y_processed']

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed
        )

        print(f"Training TabNet on {X_train.shape[0]} samples...")
        print("Note: TabNet training may take a while...")

        # Train TabNet
        tabnet = TabNetClassifierWrapper(
            max_epochs=10,  # Reduced for testing
            patience=5,
            batch_size=1024,
            seed=42,
            verbose=1
        )
        tabnet.fit(X_train, y_train)

        # Predict
        y_pred = tabnet.predict(X_test)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')

        print(f"\nTabNet Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
