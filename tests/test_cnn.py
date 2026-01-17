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


def test_tabular_dataset(synthetic_tabular_dataset):
    """Test TabularDataset creation and indexing"""
    X, y = synthetic_tabular_dataset
    
    dataset = TabularDataset(X, y)
    
    assert len(dataset) == 100, f"Dataset length should be 100 (got {len(dataset)})"
    x_item, y_item = dataset[0]
    assert x_item.shape == (10,), f"X item shape should be (10,) (got {x_item.shape})"
    assert y_item.item() in [0, 1], f"Y item should be in [0, 1] (got {y_item.item()})"
    
    # Test without labels
    dataset_no_y = TabularDataset(X, None)
    assert len(dataset_no_y) == 100, f"Dataset without labels length should be 100 (got {len(dataset_no_y)})"
    x_item_only = dataset_no_y[0]
    assert x_item_only.shape == (10,), f"X item shape should be (10,) (got {x_item_only.shape})"


def test_tabular_cnn_init_default():
    """Test TabularCNN initialization with default hidden_dims"""
    model = TabularCNN(input_dim=10, num_classes=2)
    
    assert model.input_dim == 10, f"input_dim should be 10 (got {model.input_dim})"
    assert model.num_classes == 2, f"num_classes should be 2 (got {model.num_classes})"
    assert hasattr(model, 'conv_layers'), "Model should have conv_layers attribute"
    assert hasattr(model, 'fc_layers'), "Model should have fc_layers attribute"


def test_tabular_cnn_init_custom():
    """Test TabularCNN initialization with custom hidden_dims"""
    model = TabularCNN(input_dim=20, num_classes=3, hidden_dims=[64, 32])
    
    assert model.input_dim == 20, f"input_dim should be 20 (got {model.input_dim})"
    assert model.num_classes == 3, f"num_classes should be 3 (got {model.num_classes})"
    assert len(model.conv_layers) > 0, f"conv_layers should not be empty (got {len(model.conv_layers)} layers)"


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
    
    assert output.shape == (batch_size, 2), f"Output shape should be ({batch_size}, 2) (got {output.shape})"
    assert not torch.isnan(output).any(), "Output should not contain NaN values"


def test_tabular_cnn_forward_small_input():
    """Test TabularCNN forward pass with small input dimension"""
    model = TabularCNN(input_dim=5, num_classes=2, hidden_dims=[16])
    model.eval()
    
    x = torch.randn(4, 5)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (4, 2), f"Output shape should be (4, 2) (got {output.shape})"
    assert not torch.isnan(output).any(), "Output should not contain NaN values"


def test_cnn_classifier_init():
    """Test CNNTabularClassifier initialization"""
    clf = CNNTabularClassifier(
        hidden_dims=[64, 32],
        learning_rate=0.001,
        batch_size=32,
        epochs=5,
        random_state=42
    )
    
    assert clf.hidden_dims == [64, 32], f"hidden_dims should be [64, 32] (got {clf.hidden_dims})"
    assert clf.learning_rate == 0.001, f"learning_rate should be 0.001 (got {clf.learning_rate})"
    assert clf.batch_size == 32, f"batch_size should be 32 (got {clf.batch_size})"
    assert clf.epochs == 5, f"epochs should be 5 (got {clf.epochs})"
    assert clf.random_state == 42, f"random_state should be 42 (got {clf.random_state})"
    assert clf.model is None or isinstance(clf.model, type(None)), "Model should not be initialized before fit()"


def test_cnn_classifier_init_empty_hidden_dims():
    """Test CNNTabularClassifier raises ValueError for empty hidden_dims"""
    with pytest.raises(ValueError, match="hidden_dims cannot be empty"):
        CNNTabularClassifier(hidden_dims=[])


def test_cnn_classifier_fit_predict(synthetic_binary_data):
    """Test CNNTabularClassifier fit and predict"""
    X, y = synthetic_binary_data
    n_features = X.shape[1]
    
    clf = CNNTabularClassifier(
        hidden_dims=[32, 16],
        learning_rate=0.01,
        batch_size=64,
        epochs=3,
        random_state=42
    )
    
    # Fit
    clf.fit(X, y)
    
    assert clf.model is not None, "Model should be created after fit()"
    assert clf.input_dim == n_features, f"input_dim should be {n_features} (got {clf.input_dim})"
    assert hasattr(clf.label_encoder, 'classes_'), "Label encoder should have classes_ after fit()"
    
    # Predict
    y_pred = clf.predict(X[:10])
    
    assert len(y_pred) == 10, f"Prediction length should be 10 (got {len(y_pred)})"
    assert all(pred in clf.label_encoder.classes_ for pred in y_pred), "All predictions should be in label_encoder.classes_"


def test_cnn_classifier_predict_proba(synthetic_binary_data):
    """Test CNNTabularClassifier predict_proba"""
    X, y = synthetic_binary_data
    
    clf = CNNTabularClassifier(
        hidden_dims=[32],
        epochs=3,
        random_state=42
    )
    
    clf.fit(X, y)
    
    proba = clf.predict_proba(X[:10])
    
    assert proba.shape == (10, 2), f"Probability shape should be (10, 2) (got {proba.shape})"
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5), "Probabilities should sum to 1.0 per sample"
    assert (proba >= 0).all() and (proba <= 1).all(), "All probabilities should be in [0, 1] range"


def test_cnn_classifier_multiclass(synthetic_multiclass_data):
    """Test CNNTabularClassifier with multiclass classification"""
    X, y = synthetic_multiclass_data
    
    clf = CNNTabularClassifier(
        hidden_dims=[32, 16],
        epochs=3,
        random_state=42
    )
    
    clf.fit(X, y)
    
    y_pred = clf.predict(X[:10])
    proba = clf.predict_proba(X[:10])
    
    assert len(np.unique(y)) == 3, f"Should have 3 unique classes (got {len(np.unique(y))})"
    assert proba.shape == (10, 3), f"Probability shape should be (10, 3) (got {proba.shape})"
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5), "Probabilities should sum to 1.0 per sample"


def test_cnn_classifier_sklearn_interface():
    """Test CNNTabularClassifier follows sklearn interface"""
    clf = CNNTabularClassifier(
        hidden_dims=[32],
        epochs=1,
        random_state=42
    )
    
    # Basic sklearn interface checks
    assert hasattr(clf, 'fit'), "Classifier should have fit() method"
    assert hasattr(clf, 'predict'), "Classifier should have predict() method"
    assert hasattr(clf, 'predict_proba'), "Classifier should have predict_proba() method"
    assert hasattr(clf, 'get_params'), "Classifier should have get_params() method"
    assert hasattr(clf, 'set_params'), "Classifier should have set_params() method"
