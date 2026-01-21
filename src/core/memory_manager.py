import psutil
import logging
import dask.dataframe as dd
import pandas as pd
from src.datastructure.toniot_dataframe import ToniotDataFrame
import numpy as np
import time

logger = logging.getLogger(__name__)

class MemoryAwareProcessor:
    """
    Layer 3: Support - Gestion intelligente de la mémoire.
    Calcule dynamiquement les capacités de calcul pour éviter les OOM.
    """

    def __init__(self, safety_margin: float = 0.7):
        """
        Args:
            safety_margin: Pourcentage de RAM disponible à utiliser (0.7 = 70%)
        """
        self.safety_margin = safety_margin
        logger.info(f"MemoryAwareProcessor initialized with safety_margin={safety_margin}")

    def safe_compute(self, dask_df: dd.DataFrame,
                     operation: str = "unspecified") -> ToniotDataFrame:
        """
        Convertit intelligemment Dask→Pandas selon RAM disponible avec logging verbeux.
        """
        start_time = time.time()
        
        # 1. État initial de la RAM
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024 * 1024)
        safe_mb = available_mb * self.safety_margin
        
        logger.info(f"[MEMORY] Starting operation '{operation}'")
        logger.info(f"[MEMORY] RAM Available: {available_mb:.1f}MB | Safe Limit: {safe_mb:.1f}MB ({self.safety_margin*100}%)")

        # 2. Estimation de la taille du DataFrame
        n_rows = len(dask_df)
        n_cols = len(dask_df.columns)
        
        # Estimation: 8 bytes par valeur + 20% overhead
        estimated_bytes = n_rows * n_cols * 8 * 1.2
        estimated_mb = estimated_bytes / (1024 * 1024)
        
        logger.info(f"[MEMORY] Target Data: {n_rows:,} rows x {n_cols} cols | Estimated Size: {estimated_mb:.1f}MB")

        # 3. Décision
        if estimated_mb <= safe_mb:
            logger.info(f"[MEMORY] Decision: FULL COMPUTE. RAM is sufficient.")
            result = dask_df.compute()
        else:
            safe_ratio = safe_mb / estimated_mb
            safe_rows = int(n_rows * safe_ratio)
            
            logger.warning(f"[MEMORY] Decision: SAMPLING REQUIRED! Insufficient RAM.")
            logger.warning(f"[MEMORY] Sampling Ratio: {safe_ratio*100:.2f}% | Target Rows: {safe_rows:,}")
            
            # Stratégie de sampling
            if "label" in dask_df.columns:
                logger.info("[MEMORY] Using stratified-like sampling on 'label' column")
                result = dask_df.sample(frac=safe_ratio, random_state=42).compute()
            else:
                result = dask_df.head(safe_rows)

        elapsed = time.time() - start_time
        final_mem = psutil.virtual_memory()
        logger.info(f"[MEMORY] Operation '{operation}' completed in {elapsed:.2f}s")
        logger.info(f"[MEMORY] Final RAM Usage: {final_mem.percent}%")
        
        return ToniotDataFrame(result)

    def get_memory_status(self) -> dict:
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'percent_used': mem.percent,
            'safe_available_gb': (mem.available * self.safety_margin) / (1024**3)
        }
