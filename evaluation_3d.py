#!/usr/bin/env python3
"""
3D Evaluation Framework for DDoS detection algorithms
Evaluates algorithms across three dimensions:
1. Detection Performance (F1 Score)
2. Resource Efficiency (training time, memory usage)
3. Explainability (SHAP/LIME scores)
"""
import numpy as np
import pandas as pd
import time
import psutil
import os
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import warnings

warnings.filterwarnings('ignore')

# Try to import SHAP and LIME
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class ResourceMonitor:
    """Monitor resource usage during model training"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_memory = None
        self.peak_memory = None
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start monitoring"""
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        self.start_time = time.time()
    
    def update(self):
        """Update peak memory"""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
    
    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return results"""
        self.end_time = time.time()
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'training_time_seconds': self.end_time - self.start_time,
            'memory_used_mb': self.peak_memory - self.start_memory,
            'peak_memory_mb': self.peak_memory,
            'start_memory_mb': self.start_memory
        }


class ExplainabilityEvaluator:
    """Evaluate model explainability using SHAP and LIME"""
    
    def __init__(self, feature_names: List[str] = None):
        """
        Initialize explainability evaluator
        
        Args:
            feature_names: List of feature names for explanations
        """
        self.feature_names = feature_names
        self.shap_values = None
        self.lime_explanations = None
    
    def compute_shap_score(self, model: Any, X_sample: np.ndarray, 
                          max_samples: int = 100) -> Optional[float]:
        """
        Compute SHAP explainability score
        
        Args:
            model: Trained model with predict_proba method
            X_sample: Sample data for SHAP computation
            max_samples: Maximum number of samples to use for SHAP
            
        Returns:
            SHAP explainability score (mean absolute SHAP values) or None if SHAP unavailable
        """
        if not SHAP_AVAILABLE:
            return None
        
        try:
            # Limit samples for performance
            if len(X_sample) > max_samples:
                indices = np.random.choice(len(X_sample), max_samples, replace=False)
                X_sample = X_sample[indices]
            
            # Create SHAP explainer based on model type
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model) if hasattr(model, 'estimators_') or hasattr(model, 'tree_') else shap.KernelExplainer(model.predict_proba, X_sample[:50])
            else:
                explainer = shap.KernelExplainer(model.predict, X_sample[:50])
            
            shap_values = explainer.shap_values(X_sample)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values[0]) if len(shap_values) > 0 else np.array(shap_values)
            
            # Calculate mean absolute SHAP values (higher = more explainable)
            shap_score = np.mean(np.abs(shap_values))
            
            self.shap_values = shap_values
            return float(shap_score)
        
        except Exception as e:
            print(f"Warning: SHAP computation failed: {e}")
            return None
    
    def compute_lime_score(self, model: Any, X_sample: np.ndarray, 
                          X_train: np.ndarray, max_samples: int = 10) -> Optional[float]:
        """
        Compute LIME explainability score
        
        Args:
            model: Trained model with predict_proba method
            X_sample: Sample data for LIME
            X_train: Training data for LIME background
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            LIME explainability score or None if LIME unavailable
        """
        if not LIME_AVAILABLE:
            return None
        
        try:
            if len(X_sample) > max_samples:
                indices = np.random.choice(len(X_sample), max_samples, replace=False)
                X_sample = X_sample[indices]
            
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=self.feature_names,
                mode='classification',
                discretize_continuous=True
            )
            
            lime_scores = []
            for sample in X_sample:
                explanation = explainer.explain_instance(
                    sample,
                    model.predict_proba,
                    num_features=min(10, len(sample))
                )
                # Extract importance scores
                importance = [abs(exp[1]) for exp in explanation.as_list()]
                lime_scores.append(np.mean(importance))
            
            lime_score = np.mean(lime_scores)
            return float(lime_score)
        
        except Exception as e:
            print(f"Warning: LIME computation failed: {e}")
            return None


class Evaluation3D:
    """3D evaluation framework for DDoS detection algorithms"""
    
    def __init__(self, feature_names: List[str] = None):
        """
        Initialize 3D evaluator
        
        Args:
            feature_names: List of feature names for explainability
        """
        self.feature_names = feature_names
        self.explainability_evaluator = ExplainabilityEvaluator(feature_names)
        self.results = []
    
    def evaluate_model(self, model: Any, model_name: str, 
                      X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      compute_explainability: bool = True,
                      shap_samples: int = 100,
                      lime_samples: int = 10) -> Dict:
        """
        Evaluate a model across all three dimensions
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            compute_explainability: Whether to compute explainability scores
            shap_samples: Number of samples for SHAP
            lime_samples: Number of samples for LIME
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # Dimension 1: Detection Performance
        y_pred = model.predict(X_test)
        
        # Use predict_proba if available for F1
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else model.predict_proba(X_test)
        else:
            y_pred_proba = y_pred
        
        f1 = f1_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted')
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted')
        recall = recall_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted')
        
        # Dimension 2: Resource Efficiency
        monitor = ResourceMonitor()
        monitor.start()
        
        # Retrain to measure time (or use pre-trained model if timing already done)
        model.fit(X_train, y_train)
        monitor.update()
        resource_metrics = monitor.stop()
        
        # Dimension 3: Explainability
        explainability_metrics = {}
        if compute_explainability:
            # SHAP score
            if SHAP_AVAILABLE:
                shap_score = self.explainability_evaluator.compute_shap_score(
                    model, X_test, max_samples=shap_samples
                )
                explainability_metrics['shap_score'] = shap_score
            
            # LIME score
            if LIME_AVAILABLE:
                lime_score = self.explainability_evaluator.compute_lime_score(
                    model, X_test[:lime_samples], X_train, max_samples=lime_samples
                )
                explainability_metrics['lime_score'] = lime_score
        
        # Native interpretability (for tree-based models)
        native_interpretability = 1.0 if hasattr(model, 'feature_importances_') else 0.0
        
        # Compile results
        results = {
            'model_name': model_name,
            # Dimension 1: Detection Performance
            'f1_score': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            # Dimension 2: Resource Efficiency
            'training_time_seconds': resource_metrics['training_time_seconds'],
            'memory_used_mb': resource_metrics['memory_used_mb'],
            'peak_memory_mb': resource_metrics['peak_memory_mb'],
            # Dimension 3: Explainability
            'shap_score': explainability_metrics.get('shap_score'),
            'lime_score': explainability_metrics.get('lime_score'),
            'native_interpretability': native_interpretability,
            'explainability_score': self._compute_combined_explainability(explainability_metrics, native_interpretability)
        }
        
        self.results.append(results)
        return results
    
    def _compute_combined_explainability(self, explainability_metrics: Dict, 
                                        native_interpretability: float) -> float:
        """
        Compute combined explainability score
        
        Args:
            explainability_metrics: Dictionary with SHAP and LIME scores
            native_interpretability: Binary indicator for native interpretability
            
        Returns:
            Combined explainability score (0-1 scale)
        """
        scores = []
        
        # Native interpretability (weight: 0.5)
        scores.append(native_interpretability * 0.5)
        
        # SHAP score (normalized, weight: 0.3)
        if explainability_metrics.get('shap_score') is not None:
            shap_score = explainability_metrics['shap_score']
            # Normalize (assuming typical range 0-1, adjust if needed)
            shap_norm = min(shap_score / 1.0, 1.0)  # Adjust denominator based on typical SHAP values
            scores.append(shap_norm * 0.3)
        
        # LIME score (normalized, weight: 0.2)
        if explainability_metrics.get('lime_score') is not None:
            lime_score = explainability_metrics['lime_score']
            lime_norm = min(lime_score / 1.0, 1.0)  # Adjust denominator based on typical LIME values
            scores.append(lime_norm * 0.2)
        
        return sum(scores) / sum([0.5, 0.3 if explainability_metrics.get('shap_score') else 0,
                                  0.2 if explainability_metrics.get('lime_score') else 0])
    
    def get_results_df(self) -> pd.DataFrame:
        """Get all results as a DataFrame"""
        return pd.DataFrame(self.results)
    
    def get_dimension_scores(self) -> pd.DataFrame:
        """Get normalized scores for each dimension"""
        df = self.get_results_df()
        
        # Normalize each dimension to [0, 1] scale
        dimension_scores = pd.DataFrame({
            'model_name': df['model_name'],
            'detection_performance': (df['f1_score'] - df['f1_score'].min()) / (df['f1_score'].max() - df['f1_score'].min() + 1e-10),
            'resource_efficiency': 1 - (df['training_time_seconds'] - df['training_time_seconds'].min()) / (df['training_time_seconds'].max() - df['training_time_seconds'].min() + 1e-10),
            'explainability': df['explainability_score']
        })
        
        return dimension_scores


def main():
    """Test the 3D evaluation framework"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
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
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed
        )
        
        # Initialize evaluator
        evaluator = Evaluation3D(feature_names=[f'feature_{i}' for i in range(X_processed.shape[1])])
        
        # Test models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42)  # Reduced for speed
        }
        
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            results = evaluator.evaluate_model(
                model, name, X_train, y_train, X_test, y_test,
                compute_explainability=False  # Disable for speed
            )
            print(f"  F1 Score: {results['f1_score']:.4f}")
            print(f"  Training Time: {results['training_time_seconds']:.2f}s")
        
        # Get results
        print("\n3D Evaluation Results:")
        print(evaluator.get_results_df())
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
