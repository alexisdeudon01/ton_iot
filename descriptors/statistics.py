import polars as pl
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from typing import Dict

class ColumnDescriptors:
    """
    Calcule les descripteurs statistiques exacts demandés pour une colonne.
    """
    @staticmethod
    def compute_all(series: pl.Series) -> Dict[str, float]:
        # Suppression des nulls pour les calculs statistiques
        data = series.drop_nulls().to_numpy()
        
        if len(data) == 0:
            return {
                "mean": 0.0,
                "variance": 0.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
                "entropy": 0.0
            }

        # Mean: μ = (1/N) Σ xᵢ
        mu = np.mean(data)
        
        # Variance: σ² = (1/N) Σ (xᵢ − μ)²
        var = np.var(data)
        
        # Skewness: γ₁ = E[(X−μ)³]/σ³
        # Kurtosis: γ₂ = E[(X−μ)⁴]/σ⁴ − 3
        # Utilisation de scipy pour la précision académique
        sk = skew(data)
        ku = kurtosis(data)
        
        # Entropy: H = − Σ pᵢ log(pᵢ)
        # Calcul basé sur l'histogramme des valeurs pour les données continues/discrètes
        counts = series.value_counts()
        # Extraction sécurisée des comptes
        count_col = [c for c in counts.columns if c in ["count", "counts"]][0]
        probs = counts[count_col] / counts[count_col].sum()
        ent = entropy(probs.to_numpy())

        return {
            "mean": float(mu),
            "variance": float(var),
            "skewness": float(sk),
            "kurtosis": float(ku),
            "entropy": float(ent)
        }
