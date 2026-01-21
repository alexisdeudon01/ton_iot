from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from pathlib import Path

@dataclass
class TrainingResult:
    """Résultat d'entraînement d'un modèle"""
    model_name: str
    success: bool
    training_time: float
    history: Dict[str, List[float]] = field(default_factory=dict)
    model_path: Optional[Path] = None
    error_message: Optional[str] = None

    @property
    def final_loss(self) -> float:
        """Dernière valeur de loss"""
        if not self.history or 'loss' not in self.history:
            return float('inf')
        return self.history['loss'][-1]

    @property
    def final_accuracy(self) -> float:
        """Dernière accuracy"""
        if not self.history or 'accuracy' not in self.history:
            return 0.0
        return self.history['accuracy'][-1]

@dataclass
class ValidationResult:
    """Résultat de validation hyperparamètres"""
    model_name: str
    best_params: Dict[str, Any]
    best_score: float
    all_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    tuning_plot_path: Optional[Path] = None

@dataclass
class TestResult:
    """Résultat d'évaluation finale"""
    model_name: str
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    auc: float
    report_path: Optional[Path] = None

    def to_dict(self) -> dict:
        return {
            'model': self.model_name,
            'accuracy': self.accuracy,
            'f1': self.f1_score,
            'precision': self.precision,
            'recall': self.recall,
            'auc': self.auc
        }
