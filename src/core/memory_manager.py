import psutil
import logging
import dask.dataframe as dd
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class MemoryAwareProcessor:
    """Gestion intelligente de la mémoire pour conversions Dask→Pandas"""

    def __init__(self, safety_margin: float = 0.7):
        """
        Args:
            safety_margin: Pourcentage de RAM disponible à utiliser (0.7 = 70%)
        """
        self.safety_margin = safety_margin

    def safe_compute(self, dask_df: dd.DataFrame,
                     operation: str = "training") -> pd.DataFrame:
        """
        Convertit intelligemment Dask→Pandas selon RAM disponible

        Returns:
            DataFrame pandas avec taille adaptée à la RAM
        """
        # 1. Estimer la taille en mémoire
        # Note: len(dask_df) peut être coûteux, mais nécessaire pour le sampling précis
        # On utilise une estimation rapide si possible ou on accepte le coût pour la sécurité
        try:
            n_rows = len(dask_df)
        except Exception:
            # Fallback si len échoue
            n_rows = 1000000 
            
        n_cols = len(dask_df.columns)

        # Estimation: 8 bytes par valeur numérique + 20% overhead
        estimated_bytes = n_rows * n_cols * 8 * 1.2
        estimated_mb = estimated_bytes / (1024 * 1024)

        # 2. Vérifier RAM disponible
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024 * 1024)
        safe_mb = available_mb * self.safety_margin

        logger.info(f"[MemoryAware] {operation}: Estimé={estimated_mb:.1f}MB, "
                   f"Disponible={available_mb:.1f}MB, Safe={safe_mb:.1f}MB")

        # 3. Décider: compute ou sample
        if estimated_mb <= safe_mb:
            logger.info(f"[MemoryAware] RAM suffisante, compute() complet")
            return dask_df.compute()
        else:
            # Calculer ratio de sampling sûr
            safe_ratio = safe_mb / estimated_mb
            safe_rows = int(n_rows * safe_ratio)

            logger.warning(
                f"[MemoryAware] RAM insuffisante! Sampling {safe_rows:,} rows "
                f"({safe_ratio*100:.1f}%) au lieu de {n_rows:,}"
            )

            # Échantillonnage stratifié si possible (si la colonne cible est présente)
            # On cherche 'is_ddos' ou 'label'
            target_col = None
            if "is_ddos" in dask_df.columns:
                target_col = "is_ddos"
            elif "label" in dask_df.columns:
                target_col = "label"

            if target_col:
                # Garder la distribution des classes si possible
                # Dask sample ne garantit pas la stratification exacte mais est efficace
                return dask_df.sample(frac=safe_ratio, random_state=42).compute()
            else:
                return dask_df.head(safe_rows)

    def get_memory_status(self) -> dict:
        """Retourne l'état actuel de la mémoire"""
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'percent_used': mem.percent,
            'safe_available_gb': (mem.available * self.safety_margin) / (1024**3)
        }
