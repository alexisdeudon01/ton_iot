from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class VisualizationService:
    """Service centralisé pour toutes les visualisations du pipeline"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('ggplot') # Fallback

    def plot_training_times(self, times: Dict[str, float], phase_prefix: str = "phase2") -> Path:
        """Graphique des temps d'entraînement"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(times.keys(), times.values(), color='skyblue')
        ax.set_title("Temps d'entraînement par algorithme")
        ax.set_xlabel("Algorithmes")
        ax.set_ylabel("Temps (secondes)")
        ax.grid(axis='y', alpha=0.3)

        path = self.output_dir / f"{phase_prefix}_training_times.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Graphique des temps sauvegardé: {path}")
        return path

    def plot_convergence(self, name: str, history: Dict[str, List[float]], phase_prefix: str = "phase2") -> Path:
        """Graphique de convergence (loss/accuracy)"""
        if not history or ('loss' not in history and 'accuracy' not in history):
            logger.warning(f"Pas d'historique pour {name}, plot ignoré.")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        if 'loss' in history:
            ax.plot(history['loss'], label='Loss', marker='o')
        if 'accuracy' in history:
            ax.plot(history['accuracy'], label='Accuracy', marker='s')
            
        ax.set_title(f"Convergence: {name}")
        ax.set_xlabel("Époques / Itérations")
        ax.set_ylabel("Valeur")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = self.output_dir / f"{phase_prefix}_convergence_{name.lower()}.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Graphique de convergence sauvegardé: {path}")
        return path

    def plot_tuning_results(self, name: str, param_name: str, values: List[Any], 
                            accs: List[float], f1s: List[float], aucs: List[float],
                            phase_prefix: str = "phase3") -> Path:
        """Graphique des résultats de tuning hyperparamètres"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(values, accs, label='Accuracy', marker='o')
        ax.plot(values, f1s, label='F1-Score', marker='s')
        ax.plot(values, aucs, label='AUC', marker='^')
        
        ax.set_title(f"Validation Tuning: {name} ({param_name})")
        ax.set_xlabel(f"Variation de {param_name}")
        ax.set_ylabel("Scores")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = self.output_dir / f"{phase_prefix}_tuning_{name.lower()}.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Graphique de tuning sauvegardé: {path}")
        return path

    def plot_final_metrics(self, test_results: Dict[str, Dict[str, float]], 
                           metrics_names: List[str], phase_prefix: str = "phase5") -> Path:
        """Synthèse des performances finales"""
        if not test_results:
            return None

        fig, ax = plt.subplots(figsize=(12, 7))
        algos = list(test_results.keys())

        for metric in metrics_names:
            scores = [test_results[algo].get(metric, 0) for algo in algos]
            ax.plot(algos, scores, marker='o', label=metric, linewidth=2)

        ax.set_title("Synthèse des Performances Finales (Comparaison des Algos)")
        ax.set_xlabel("Algorithmes")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        path = self.output_dir / f"{phase_prefix}_final_synthesis.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Graphique de synthèse finale sauvegardé: {path}")
        return path

    def plot_xai_metrics(self, results: Dict[str, Dict[str, Dict[str, float]]], 
                         phase_prefix: str = "phase4") -> List[Path]:
        """Graphiques des métriques XAI"""
        paths = []
        for metric in ['fidelity', 'stability', 'complexity']:
            fig, ax = plt.subplots(figsize=(10, 6))
            for algo in results:
                methods = list(results[algo].keys())
                scores = [results[algo][m].get(metric, 0) for m in methods]
                ax.plot(methods, scores, marker='o', label=algo)

            ax.set_title(f"XAI Validation: {metric.capitalize()}")
            ax.set_xlabel("Méthode XAI")
            ax.set_ylabel("Score")
            ax.legend()
            ax.grid(True, alpha=0.3)

            path = self.output_dir / f"{phase_prefix}_xai_{metric}.png"
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            paths.append(path)
            
        return paths
