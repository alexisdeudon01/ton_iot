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
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Setup logger
logger = logging.getLogger(__name__)

# Visualizations are handled by src/evaluation/visualizations.py (matplotlib only, no seaborn)
VISUALIZATION_AVAILABLE = True  # Assume matplotlib is available (core dependency)

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
        self.use_tracemalloc = False
        self.tracemalloc_current = None
        self.tracemalloc_peak = None
        
        # Try to use tracemalloc for more accurate peak memory tracking
        try:
            import tracemalloc
            self.use_tracemalloc = True
            self.tracemalloc_module = tracemalloc
        except ImportError:
            self.use_tracemalloc = False
    
    def start(self):
        """Start monitoring"""
        self.start_time = time.perf_counter()
        
        if self.use_tracemalloc:
            self.tracemalloc_module.start()
            self.tracemalloc_current = self.tracemalloc_module.take_snapshot()
        
        # Fallback to psutil
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
    
    def update(self):
        """Update peak memory"""
        if self.use_tracemalloc:
            current_snapshot = self.tracemalloc_module.take_snapshot()
            current_size = sum(stat.size for stat in current_snapshot.statistics('lineno')) / 1024 / 1024  # MB
            if self.tracemalloc_peak is None or current_size > self.tracemalloc_peak:
                self.tracemalloc_peak = current_size
        
        # Also update psutil peak (fallback)
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
    
    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return results"""
        self.end_time = time.perf_counter()
        
        # Get peak memory (prefer tracemalloc if available)
        if self.use_tracemalloc:
            try:
                final_snapshot = self.tracemalloc_module.take_snapshot()
                final_size = sum(stat.size for stat in final_snapshot.statistics('lineno')) / 1024 / 1024  # MB
                peak_memory_mb = max(self.tracemalloc_peak or 0, final_size)
                self.tracemalloc_module.stop()
            except Exception:
                # Fallback to psutil if tracemalloc fails
                peak_memory_mb = self.peak_memory
        else:
            peak_memory_mb = self.peak_memory
        
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'training_time_seconds': self.end_time - self.start_time,
            'memory_used_mb': peak_memory_mb - self.start_memory,
            'peak_memory_mb': peak_memory_mb,
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
        # Dimension 2: Resource Efficiency - Train model first and measure resources
        monitor = ResourceMonitor()
        monitor.start()
        
        # Create a fresh, unfitted model instance to avoid state issues across folds
        from src.core.model_utils import fresh_model
        model_clone = fresh_model(model)
        
        # Train the model
        model_clone.fit(X_train, y_train)
        monitor.update()
        resource_metrics = monitor.stop()
        
        # Dimension 1: Detection Performance - Now that model is trained
        y_pred = model_clone.predict(X_test)
        
        # Use predict_proba if available
        if hasattr(model_clone, 'predict_proba'):
            y_pred_proba = model_clone.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else model_clone.predict_proba(X_test)
        else:
            y_pred_proba = y_pred
        
        # Dimension 1: Detection Performance Metrics
        # Following CIC-DDoS2019 methodology: Precision (Pr), Recall (Rc), F1 Score
        # Using weighted average for multi-class problems as per CIC-DDoS2019 paper
        # Reference: "Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy"
        is_binary = len(np.unique(y_test)) == 2
        avg_method = 'binary' if is_binary else 'weighted'
        
        f1 = f1_score(y_test, y_pred, average=avg_method)  # F1 Score (primary metric for Dimension 1)
        accuracy = accuracy_score(y_test, y_pred)  # Overall accuracy
        precision = precision_score(y_test, y_pred, average=avg_method)  # Precision (Pr) from CIC-DDoS2019
        recall = recall_score(y_test, y_pred, average=avg_method)  # Recall (Rc) from CIC-DDoS2019
        
        # Confusion Matrix - Detailed calculation
        cm = confusion_matrix(y_test, y_pred)
        logger.debug(f"Confusion Matrix for {model_name}: {cm.tolist()}")
        
        # Extract TP, TN, FP, FN for binary classification
        if is_binary and cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            cm_details = {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'total_samples': int(cm.sum())
            }
        else:
            # Multi-class: compute total TP, TN, FP, FN
            tn = cm.diagonal().sum()
            fp = cm.sum(axis=0) - cm.diagonal()
            fn = cm.sum(axis=1) - cm.diagonal()
            tp = cm.diagonal()
            cm_details = {
                'confusion_matrix': cm.tolist(),
                'total_samples': int(cm.sum()),
                'class_wise_tp': tp.tolist() if hasattr(tp, 'tolist') else list(tp),
                'class_wise_fp': fp.tolist() if hasattr(fp, 'tolist') else list(fp),
                'class_wise_fn': fn.tolist() if hasattr(fn, 'tolist') else list(fn),
            }
        
        # Measure inference latency (with warm-up for NN models)
        is_nn = 'CNN' in model_name or 'TabNet' in model_name
        warmup_runs = 2 if is_nn else 0
        latency_runs = 50 if is_nn else 100  # Reduce for NN (slower)
        
        # Warm-up for NN models (avoid noise from first inference)
        if warmup_runs > 0 and len(X_test) > 0:
            _ = model_clone.predict(X_test[:min(100, len(X_test))])
        
        # Measure inference latency (N runs)
        if len(X_test) > 0:
            latency_subset = X_test[:min(100, len(X_test))]
            latency_start = time.perf_counter()
            for _ in range(latency_runs):
                _ = model_clone.predict(latency_subset)
            latency_end = time.perf_counter()
            inference_latency_ms = ((latency_end - latency_start) / latency_runs) * 1000
        else:
            inference_latency_ms = 0.0
        
        # Dimension 3: Explainability
        explainability_metrics = {}
        if compute_explainability:
            # SHAP score (use trained model_clone)
            if SHAP_AVAILABLE:
                shap_score = self.explainability_evaluator.compute_shap_score(
                    model_clone, X_test, max_samples=shap_samples
                )
                explainability_metrics['shap_score'] = shap_score
            
            # LIME score (use trained model_clone)
            if LIME_AVAILABLE:
                lime_score = self.explainability_evaluator.compute_lime_score(
                    model_clone, X_test[:lime_samples], X_train, max_samples=lime_samples
                )
                explainability_metrics['lime_score'] = lime_score
        
        # Native interpretability (for tree-based models and LR - use trained model_clone)
        if hasattr(model_clone, 'feature_importances_'):
            native_interpretability = 1.0  # DT/RF
        elif hasattr(model_clone, 'coef_'):
            native_interpretability = 1.0  # LR
        else:
            native_interpretability = 0.0  # CNN/TabNet
        
        # Compile results with detailed confusion matrix information
        results = {
            'model_name': model_name,
            # Dimension 1: Detection Performance
            'f1_score': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm.tolist() if hasattr(cm, 'tolist') else cm,
            'confusion_matrix_details': cm_details,
            # Dimension 2: Resource Efficiency
            'training_time_seconds': resource_metrics['training_time_seconds'],
            'memory_used_mb': resource_metrics['memory_used_mb'],
            'peak_memory_mb': resource_metrics['peak_memory_mb'],
            'inference_latency_ms': inference_latency_ms,
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
        """
        Get normalized scores for each dimension according to IRP methodology
        
        Formulas according to IRP_FinalADE_v2.0ADE-2-1.pdf:
        - Dimension 1: F1 Score (normalized min-max)
        - Dimension 2: 0.6 * normalized_time + 0.4 * normalized_memory (inverse normalized)
        - Dimension 3: 0.5 * native + 0.3 * normalized_SHAP + 0.2 * normalized_LIME
        """
        df = self.get_results_df()
        
        # Dimension 1: Detection Performance (F1 Score normalized)
        f1_min = df['f1_score'].min()
        f1_max = df['f1_score'].max()
        f1_range = f1_max - f1_min + 1e-10  # Avoid division by zero
        dimension1_normalized = (df['f1_score'] - f1_min) / f1_range
        
        # Dimension 2: Resource Efficiency (0.6 * time + 0.4 * memory)
        # Normalize time and memory (inverse: less is better -> more is better)
        time_min = df['training_time_seconds'].min()
        time_max = df['training_time_seconds'].max()
        time_range = time_max - time_min + 1e-10
        normalized_time = 1 - (df['training_time_seconds'] - time_min) / time_range
        
        memory_min = df['memory_used_mb'].min()
        memory_max = df['memory_used_mb'].max()
        memory_range = memory_max - memory_min + 1e-10
        normalized_memory = 1 - (df['memory_used_mb'] - memory_min) / memory_range
        
        # Combined Dimension 2: 0.6 * time + 0.4 * memory (IRP formula)
        dimension2_score = 0.6 * normalized_time + 0.4 * normalized_memory
        
        # Dimension 3: Explainability (already normalized in _compute_combined_explainability)
        # Using: 0.5 * native + 0.3 * normalized_SHAP + 0.2 * normalized_LIME
        dimension3_normalized = df['explainability_score']
        
        # Create normalized dimension scores DataFrame
        dimension_scores = pd.DataFrame({
            'model_name': df['model_name'],
            'detection_performance': dimension1_normalized,  # F1 Score normalized
            'resource_efficiency': dimension2_score,  # 0.6*time + 0.4*memory
            'explainability': dimension3_normalized  # Already normalized (0.5*native + 0.3*SHAP + 0.2*LIME)
        })
        
        return dimension_scores
    
    def generate_algorithm_report(self, model_name: str, results: Dict, output_dir: Path) -> str:
        """
        Generate a detailed report for an algorithm explaining the 3 dimensions
        
        Args:
            model_name: Name of the algorithm
            results: Dictionary with evaluation results
            output_dir: Directory to save the report
            
        Returns:
            Path to saved report file
        """
        report_path = output_dir / 'algorithm_reports' / f'{model_name.replace(" ", "_")}_report.md'
        
        report = f"# Rapport d'Évaluation : {model_name}\n\n"
        report += "## Résumé\n\n"
        report += f"- **F1 Score**: {results.get('f1_score', 0):.4f}\n"
        report += f"- **Resource Efficiency Score**: {self._calculate_resource_efficiency(results, self.results):.4f}\n"
        report += f"- **Explainability Score**: {results.get('explainability_score', 0):.4f}\n\n"
        
        # Dimension 1
        report += "## Dimension 1: Detection Performance\n\n"
        report += f"- **F1 Score**: {results.get('f1_score', 0):.4f} (métrique principale)\n"
        report += f"- **Precision (Pr)**: {results.get('precision', 0):.4f}\n"
        report += f"- **Recall (Rc)**: {results.get('recall', 0):.4f}\n"
        report += f"- **Accuracy**: {results.get('accuracy', 0):.4f}\n\n"
        
        f1 = results.get('f1_score', 0)
        if f1 > 0.9:
            perf_interp = "Excellente performance de détection"
        elif f1 > 0.7:
            perf_interp = "Bonne performance de détection"
        elif f1 > 0.5:
            perf_interp = "Performance acceptable"
        else:
            perf_interp = "Performance insuffisante"
        
        report += f"**Interprétation**: {perf_interp}. Le modèle a un bon équilibre entre Precision et Recall.\n\n"
        
        # Dimension 2
        report += "## Dimension 2: Resource Efficiency\n\n"
        report += f"- **Training Time**: {results.get('training_time_seconds', 0):.2f} seconds\n"
        report += f"- **Memory Usage**: {results.get('memory_used_mb', 0):.2f} MB\n"
        report += f"- **Peak Memory**: {results.get('peak_memory_mb', 0):.2f} MB\n\n"
        
        time_score = self._calculate_resource_efficiency(results, self.results)
        if time_score > 0.8:
            eff_interp = "Très efficace (rapide et peu de mémoire)"
        elif time_score > 0.5:
            eff_interp = "Efficace"
        elif time_score > 0.3:
            eff_interp = "Efficacité modérée"
        else:
            eff_interp = "Peu efficace (lent ou gourmand en mémoire)"
        
        report += f"**Interprétation**: {eff_interp}.\n\n"
        
        # Dimension 3
        report += "## Dimension 3: Explainability\n\n"
        report += f"- **Native Interpretability**: {results.get('native_interpretability', 0):.1f} "
        if results.get('native_interpretability', 0) == 1.0:
            report += "(Modèle interprétable nativement - tree-based)\n"
        else:
            report += "(Pas d'interprétabilité native - boîte noire)\n"
        report += f"- **SHAP Score**: {results.get('shap_score', 'N/A')}\n"
        report += f"- **LIME Score**: {results.get('lime_score', 'N/A')}\n"
        report += f"- **Combined Explainability Score**: {results.get('explainability_score', 0):.4f}\n\n"
        
        exp_score = results.get('explainability_score', 0)
        if exp_score > 0.7:
            exp_interp = "Très explicable"
        elif exp_score > 0.4:
            exp_interp = "Modérément explicable"
        else:
            exp_interp = "Peu explicable (boîte noire)"
        
        report += f"**Interprétation**: {exp_interp}.\n\n"
        
        # Strengths and weaknesses
        report += "## Forces et Faiblesses\n\n"
        strengths = []
        weaknesses = []
        
        if f1 > 0.85:
            strengths.append("Excellente performance de détection (F1 > 0.85)")
        if time_score > 0.7:
            strengths.append("Entraînement rapide et efficace en mémoire")
        if exp_score > 0.6:
            strengths.append("Modèle interprétable")
        
        if f1 < 0.7:
            weaknesses.append("Performance de détection modérée (F1 < 0.7)")
        if time_score < 0.4:
            weaknesses.append("Entraînement lent ou gourmand en ressources")
        if exp_score < 0.3:
            weaknesses.append("Faible explicabilité")
        
        if strengths:
            report += "**Forces**:\n"
            for s in strengths:
                report += f"- {s}\n"
            report += "\n"
        
        if weaknesses:
            report += "**Faiblesses**:\n"
            for w in weaknesses:
                report += f"- {w}\n"
            report += "\n"
        
        # Save report
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Generated algorithm report: {report_path}")
            return str(report_path)
        except Exception as e:
            logger.error(f"Error saving algorithm report: {e}", exc_info=True)
            return None
    
    def _calculate_resource_efficiency(self, results: Dict, all_results: Optional[List[Dict]] = None) -> float:
        """
        Calculate normalized resource efficiency score for a single model
        
        Formula: 0.6 * normalized_time + 0.4 * normalized_memory (inverse normalized)
        Higher score = more efficient (faster and/or less memory)
        
        Args:
            results: Results dict for the current model
            all_results: Optional list of all results for proper normalization
            
        Returns:
            Resource efficiency score (0-1, higher is better)
        """
        # If all_results provided, use for normalization
        if all_results and len(all_results) > 1:
            all_times = [r.get('training_time_seconds', 0) for r in all_results]
            all_mems = [r.get('memory_used_mb', 0) for r in all_results]
            time_min, time_max = min(all_times), max(all_times)
            mem_min, mem_max = min(all_mems), max(all_mems)
        else:
            # Single model: use self.results if available, otherwise use values directly
            if self.results and len(self.results) > 1:
                df_temp = pd.DataFrame(self.results)
                time_min, time_max = df_temp['training_time_seconds'].min(), df_temp['training_time_seconds'].max()
                mem_min, mem_max = df_temp['memory_used_mb'].min(), df_temp['memory_used_mb'].max()
            else:
                # Fallback: assume reasonable ranges if only one model
                time_val = results.get('training_time_seconds', 0)
                mem_val = results.get('memory_used_mb', 0)
                time_min, time_max = 0, max(time_val * 2, 1)
                mem_min, mem_max = 0, max(mem_val * 2, 1)
        
        # Normalize time and memory (inverse: less is better -> more is better)
        time_val = results.get('training_time_seconds', 0)
        mem_val = results.get('memory_used_mb', 0)
        
        time_range = time_max - time_min + 1e-10
        normalized_time = 1 - (time_val - time_min) / time_range
        
        mem_range = mem_max - mem_min + 1e-10
        normalized_memory = 1 - (mem_val - mem_min) / mem_range
        
        # Combined Dimension 2: 0.6 * time + 0.4 * memory (IRP formula)
        resource_efficiency = 0.6 * normalized_time + 0.4 * normalized_memory
        
        return float(resource_efficiency)
    
    def generate_dimension_visualizations(self, output_dir: Path) -> Dict[str, str]:
        """
        Generate visualizations for each of the 3 dimensions using centralized module
        
        Uses src/evaluation/visualizations.py (matplotlib only, no seaborn)
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        try:
            from src.evaluation.visualizations import generate_all_visualizations
        except ImportError:
            logger.warning("Visualization module not available - skipping visualizations")
            return {}
        
        df = self.get_results_df()
        if df.empty:
            logger.warning("No results available for visualization")
            return {}
        
        # Prepare data for visualizations
        try:
            # Get per-fold metrics if available
            metrics_by_fold_df = None
            if hasattr(self, 'fold_results') and self.fold_results:
                metrics_by_fold_df = pd.DataFrame(self.fold_results)
            
            # Get normalized scores if available
            scores_normalized_df = None
            if hasattr(self, 'normalized_scores') and self.normalized_scores:
                scores_normalized_df = pd.DataFrame(self.normalized_scores)
            
            # Confusion matrices, ROC/PR curves (if available)
            confusion_matrices = {}
            roc_curves = {}
            pr_curves = {}
            
            # SHAP/LIME values (if available)
            shap_values_dict = {}
            lime_importances_dict = {}
            feature_names = None
            
            # Call centralized visualization generator
            saved_files = generate_all_visualizations(
                metrics_df=df,
                metrics_by_fold_df=metrics_by_fold_df,
                scores_normalized_df=scores_normalized_df,
                confusion_matrices=confusion_matrices if confusion_matrices else None,
                roc_curves=roc_curves if roc_curves else None,
                pr_curves=pr_curves if pr_curves else None,
                shap_values_dict=shap_values_dict if shap_values_dict else None,
                lime_importances_dict=lime_importances_dict if lime_importances_dict else None,
                feature_names=feature_names,
                output_dir=output_dir
            )
            
            # Convert Path objects to strings for compatibility
            return {k: str(v) for k, v in saved_files.items()}
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}", exc_info=True)
            return {}


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
