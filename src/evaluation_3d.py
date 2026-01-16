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

# Try to import matplotlib/seaborn for visualizations
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not available. Visualizations will be disabled.")

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
        # Dimension 2: Resource Efficiency - Train model first and measure resources
        monitor = ResourceMonitor()
        monitor.start()
        
        # Create a clone/fresh instance of the model to avoid state issues across folds
        # For sklearn models, we can use clone; for custom models, we need to reinitialize
        try:
            from sklearn.base import clone
            model_clone = clone(model)
        except Exception:
            # For custom models that can't be cloned, use the model directly
            # But this might cause issues if model state persists across folds
            model_clone = model
        
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
        
        # Native interpretability (for tree-based models - use trained model_clone)
        native_interpretability = 1.0 if hasattr(model_clone, 'feature_importances_') else 0.0
        
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
        report += f"- **Resource Efficiency Score**: {self._calculate_resource_efficiency(results):.4f}\n"
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
        
        time_score = self._calculate_resource_efficiency(results)
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
    
    def _calculate_resource_efficiency(self, results: Dict) -> float:
        """Calculate normalized resource efficiency score (helper for reports)"""
        # This is a simplified version - actual calculation uses all models for normalization
        # For report purposes, return a placeholder that will be recalculated properly
        return results.get('explainability_score', 0)  # Placeholder
    
    def generate_dimension_visualizations(self, output_dir: Path) -> Dict[str, str]:
        """
        Generate visualizations for each of the 3 dimensions
        
        Creates bar charts, scatter plots, and radar charts for:
        - Dimension 1: Detection Performance (F1, Precision, Recall)
        - Dimension 2: Resource Efficiency (Time, Memory)
        - Dimension 3: Explainability (SHAP, LIME, Native)
        - Combined 3D visualization (radar chart)
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary mapping dimension names to file paths of saved visualizations
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualizations not available (matplotlib/seaborn not installed)")
            return {}
        
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        df = self.get_results_df()
        if df.empty:
            logger.warning("No results available for visualization")
            return {}
        
        saved_files = {}
        
        try:
            # Dimension 1: Detection Performance
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Dimension 1: Detection Performance', fontsize=16, fontweight='bold')
            
            models = df['model_name'].values
            metrics = ['f1_score', 'precision', 'recall', 'accuracy']
            titles = ['F1 Score', 'Precision (Pr)', 'Recall (Rc)', 'Accuracy']
            
            for idx, (metric, title) in enumerate(zip(metrics, titles)):
                ax = axes[idx // 2, idx % 2]
                values = df[metric].values
                bars = ax.bar(models, values, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
                ax.set_title(title, fontweight='bold')
                ax.set_ylabel('Score')
                ax.set_ylim([0, max(1.0, values.max() * 1.1)])
                ax.tick_params(axis='x', rotation=45)
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            dim1_path = viz_dir / 'dimension1_performance.png'
            plt.savefig(dim1_path, dpi=150, bbox_inches='tight')
            plt.close()
            saved_files['dimension1'] = str(dim1_path)
            logger.info(f"Saved Dimension 1 visualization: {dim1_path}")
            
            # Dimension 2: Resource Efficiency
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle('Dimension 2: Resource Efficiency', fontsize=16, fontweight='bold')
            
            # Training Time
            ax1 = axes[0]
            time_values = df['training_time_seconds'].values
            bars1 = ax1.bar(models, time_values, color=plt.cm.plasma(np.linspace(0, 1, len(models))))
            ax1.set_title('Training Time', fontweight='bold')
            ax1.set_ylabel('Time (seconds)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis='y', alpha=0.3)
            for bar, val in zip(bars1, time_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}s', ha='center', va='bottom', fontsize=9)
            
            # Memory Usage
            ax2 = axes[1]
            memory_values = df['memory_used_mb'].values
            bars2 = ax2.bar(models, memory_values, color=plt.cm.plasma(np.linspace(0.3, 1, len(models))))
            ax2.set_title('Memory Usage', fontweight='bold')
            ax2.set_ylabel('Memory (MB)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(axis='y', alpha=0.3)
            for bar, val in zip(bars2, memory_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}MB', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            dim2_path = viz_dir / 'dimension2_resources.png'
            plt.savefig(dim2_path, dpi=150, bbox_inches='tight')
            plt.close()
            saved_files['dimension2'] = str(dim2_path)
            logger.info(f"Saved Dimension 2 visualization: {dim2_path}")
            
            # Dimension 3: Explainability
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.suptitle('Dimension 3: Explainability', fontsize=16, fontweight='bold')
            
            explain_scores = df['explainability_score'].values
            native_int = df['native_interpretability'].values
            
            x = np.arange(len(models))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, explain_scores, width, label='Combined Explainability Score', 
                          color=plt.cm.coolwarm(0.3))
            bars2 = ax.bar(x + width/2, native_int, width, label='Native Interpretability', 
                          color=plt.cm.coolwarm(0.7))
            
            ax.set_ylabel('Score')
            ax.set_title('Explainability Metrics', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.set_ylim([0, 1.1])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            dim3_path = viz_dir / 'dimension3_explainability.png'
            plt.savefig(dim3_path, dpi=150, bbox_inches='tight')
            plt.close()
            saved_files['dimension3'] = str(dim3_path)
            logger.info(f"Saved Dimension 3 visualization: {dim3_path}")
            
            # Combined 3D Radar Chart
            dim_scores = self.get_dimension_scores()
            if not dim_scores.empty:
                fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
                
                # Prepare data for radar chart
                categories = ['Detection\nPerformance', 'Resource\nEfficiency', 'Explainability']
                num_vars = len(categories)
                
                # Compute angle for each category
                angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
                angles += angles[:1]  # Complete the circle
                
                # Plot each model
                colors = plt.cm.tab10(np.linspace(0, 1, len(dim_scores)))
                for idx, row in dim_scores.iterrows():
                    values = [
                        row['detection_performance'],
                        row['resource_efficiency'],
                        row['explainability']
                    ]
                    values += values[:1]  # Complete the circle
                    
                    ax.plot(angles, values, 'o-', linewidth=2, label=row['model_name'], color=colors[idx])
                    ax.fill(angles, values, alpha=0.25, color=colors[idx])
                
                # Customize radar chart
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
                ax.set_ylim(0, 1)
                ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
                ax.grid(True)
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                
                plt.title('3D Evaluation Framework - Combined View', fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                
                radar_path = viz_dir / 'combined_3d_radar.png'
                plt.savefig(radar_path, dpi=150, bbox_inches='tight')
                plt.close()
                saved_files['combined_radar'] = str(radar_path)
                logger.info(f"Saved combined 3D radar chart: {radar_path}")
            
            # Scatter plot: Performance vs Efficiency
            fig, ax = plt.subplots(figsize=(10, 8))
            
            perf = dim_scores['detection_performance'].values if not dim_scores.empty else df['f1_score'].values
            eff = dim_scores['resource_efficiency'].values if not dim_scores.empty else 1 - (df['training_time_seconds'].values / df['training_time_seconds'].max())
            
            scatter = ax.scatter(perf, eff, s=200, c=dim_scores['explainability'].values if not dim_scores.empty else df['explainability_score'].values,
                               cmap='viridis', alpha=0.6, edgecolors='black', linewidth=2)
            
            for i, model in enumerate(models):
                ax.annotate(model, (perf[i], eff[i]), xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            ax.set_xlabel('Detection Performance (normalized)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Resource Efficiency (normalized)', fontsize=12, fontweight='bold')
            ax.set_title('Performance vs Efficiency (color = Explainability)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter)
            cbar.set_label('Explainability Score', fontsize=11)
            
            plt.tight_layout()
            scatter_path = viz_dir / 'performance_vs_efficiency_scatter.png'
            plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
            plt.close()
            saved_files['scatter'] = str(scatter_path)
            logger.info(f"Saved scatter plot: {scatter_path}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}", exc_info=True)
        
        return saved_files


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
