#!/usr/bin/env python3
"""
Configuration pour Phase 3: Evaluation 3D
Poids, seuils, paramètres pour les calculs des 3 dimensions
"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Phase3Config:
    """Configuration pour Phase 3 Evaluation"""
    
    # Dimension 2: Resource Efficiency weights
    resource_weights: Dict[str, float] = None  # w_time, w_ram, w_latency
    
    # Dimension 3: Explainability weights
    explainability_weights: Dict[str, float] = None  # w_native, w_shap, w_lime
    
    # Latency measurement
    latency_n_runs: int = 100  # Nombre de runs pour mesurer latency
    
    # Explainability
    shap_sample_size: int = 100
    lime_sample_size: int = 10
    top_k_features: int = 10  # Top-k features pour SHAP/LIME
    
    # Output paths
    metrics_dir: str = "output/phase3_evaluation/metrics"
    visualizations_dir: str = "output/phase3_evaluation/visualizations"
    reports_dir: str = "output/phase3_evaluation/algorithm_reports"
    
    def __post_init__(self):
        """Set defaults if None"""
        if self.resource_weights is None:
            self.resource_weights = {
                'w_time': 0.5,
                'w_ram': 0.3,
                'w_latency': 0.2
            }
        
        if self.explainability_weights is None:
            self.explainability_weights = {
                'w_native': 0.5,
                'w_shap': 0.3,
                'w_lime': 0.2
            }
        
        # Validate weights sum to 1.0
        if abs(sum(self.resource_weights.values()) - 1.0) > 1e-6:
            raise ValueError(f"Resource weights must sum to 1.0, got {sum(self.resource_weights.values())}")
        
        if abs(sum(self.explainability_weights.values()) - 1.0) > 1e-6:
            raise ValueError(f"Explainability weights must sum to 1.0, got {sum(self.explainability_weights.values())}")
