"""
Tests for cnn.py - TabularCNN and CNNTabularClassifier
"""
import sys
from pathlib import Path
import pytest
import numpy as np

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

pytest.importorskip("torch", reason="torch not available")

from src.models.cnn import TabularCNN, CNNTabularClassifier, TabularDataset
import torch


def test_tabular_dataset():
    """Test TabularDataset creation and indexing"""
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randint(0, 2, 100).astype(np.int64)
    
    dataset = TabularDataset(X, y)
    
    assert len(dataset) == 100
    x_item, y_item = dataset[0]
    assert x_item.shape == (10,)
    assert y_item.item() in [0, 1]
    
    # Test without labels
    dataset_no_y = TabularDataset(X, None)
    assert len(dataset_no_y) == 100
    x_item_only = dataset_no_y[0]
    assert x_item_only.shape == (10,)


def test_tabular_cnn_init_default():
    """Test TabularCNN initialization with default hidden_dims"""
    model = TabularCNN(input_dim=10, num_classes=2)
    
    assert model.input_dim == 10
    assert model.num_classes == 2
    assert hasattr(model, 'conv_layers')
    assert hasattr(model, 'fc_layers')


def test_tabular_cnn_init_custom():
    """Test TabularCNN initialization with custom hidden_dims"""
    model = TabularCNN(input_dim=20, num_classes=3, hidden_dims=[64, 32])
    
    assert model.input_dim == 20
    assert model.num_classes == 3
    assert len(model.conv_layers) > 0


def test_tabular_cnn_init_empty_hidden_dims():
    """Test TabularCNN raises ValueError for empty hidden_dims"""
    with pytest.raises(ValueError, match="hidden_dims cannot be empty"):
        TabularCNN(input_dim=10, num_classes=2, hidden_dims=[])


def test_tabular_cnn_forward():
    """Test TabularCNN forward pass"""
    model = TabularCNN(input_dim=10, num_classes=2, hidden_dims=[32, 16])
    model.eval()
    
    batch_size = 8
    x = torch.randn(batch_size, 10)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (batch_size, 2)
    assert not torch.isnan(output).any()


def test_tabular_cnn_forward_small_input():
    """Test TabularCNN forward pass with small input dimension"""
    model = TabularCNN(input_dim=5, num_classes=2, hidden_dims=[16])
    model.eval()
    
    x = torch.randn(4, 5)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (4, 2)
    assert not torch.isnan(output).any()


def test_cnn_classifier_init():
    """Test CNNTabularClassifier initialization"""
    clf = CNNTabularClassifier(
        hidden_dims=[64, 32],
        learning_rate=0.001,
        batch_size=32,
        epochs=5,
        random_state=42
    )
    
    assert clf.hidden_dims == [64, 32]
    assert clf.learning_rate == 0.001
    assert clf.batch_size == 32
    assert clf.epochs == 5
    assert clf.random_state == 42
    assert clf.model is None or isinstance(clf.model, type(None))


def test_cnn_classifier_init_empty_hidden_dims():
    """Test CNNTabularClassifier raises ValueError for empty hidden_dims"""
    with pytest.raises(ValueError, match="hidden_dims cannot be empty"):
        CNNTabularClassifier(hidden_dims=[])


def test_cnn_classifier_fit_predict():
    """Test CNNTabularClassifier fit and predict"""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples)
    
    clf = CNNTabularClassifier(
        hidden_dims=[32, 16],
        learning_rate=0.01,
        batch_size=64,
        epochs=3,
        random_state=42
    )
    
    # Fit
    clf.fit(X, y)
    
    assert clf.model is not None
    assert clf.input_dim == n_features
    assert hasattr(clf.label_encoder, 'classes_')
    
    # Predict
    y_pred = clf.predict(X[:10])
    
    assert len(y_pred) == 10
    assert all(pred in clf.label_encoder.classes_ for pred in y_pred)


def test_cnn_classifier_predict_proba():
    """Test CNNTabularClassifier predict_proba"""
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples)
    
    clf = CNNTabularClassifier(
        hidden_dims=[32],
        epochs=3,
        random_state=42
    )
    
    clf.fit(X, y)
    
    proba = clf.predict_proba(X[:10])
    
    assert proba.shape == (10, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_cnn_classifier_multiclass():
    """Test CNNTabularClassifier with multiclass classification"""
    np.random.seed(42)
    n_samples = 300
    n_features = 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 3, n_samples)  # 3 classes
    
    clf = CNNTabularClassifier(
        hidden_dims=[32, 16],
        epochs=3,
        random_state=42
    )
    
    clf.fit(X, y)
    
    y_pred = clf.predict(X[:10])
    proba = clf.predict_proba(X[:10])
    
    assert len(np.unique(y)) == 3
    assert proba.shape == (10, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_cnn_classifier_sklearn_interface():
    """Test CNNTabularClassifier follows sklearn interface"""
    from sklearn.utils.estimator_checks import check_estimator
    
    clf = CNNTabularClassifier(
        hidden_dims=[32],
        epochs=1,
        random_state=42
    )
    
    # Basic sklearn interface checks
    assert hasattr(clf, 'fit')
    assert hasattr(clf, 'predict')
    assert hasattr(clf, 'predict_proba')
    assert hasattr(clf, 'get_params')
    assert hasattr(clf, 'set_params')
