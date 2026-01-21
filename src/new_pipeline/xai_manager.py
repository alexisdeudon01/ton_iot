import logging
import time
import psutil
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.inspection import permutation_importance

from src.new_pipeline.config import config

logger = logging.getLogger(__name__)

class LateFusionXAI:
    """Phase 7: Ressources & XAI (Global & Local)."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.p8 = config.phase8
        self.p7 = config.phase7
        logger.info(f"Initialized LateFusionXAI. Artifacts in {output_dir}")

    def run_resource_audit(self, model, X_sample: pd.DataFrame) -> Dict[str, float]:
        """Measures inference latency, CPU usage, and peak memory."""
        logger.info(f"Auditing resources for model on {len(X_sample)} samples...")
        
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)
        cpu_before = process.cpu_percent(interval=None)
        
        start_time = time.time()
        _ = model.predict(X_sample)
        end_time = time.time()
        
        cpu_after = process.cpu_percent(interval=None)
        mem_after = process.memory_info().rss / (1024 * 1024)
        
        latency = (end_time - start_time) / len(X_sample)
        
        audit = {
            'inference_latency_ms': latency * 1000,
            'peak_memory_mb': mem_after,
            'memory_delta_mb': mem_after - mem_before,
            'cpu_usage_percent': cpu_after
        }
        logger.info(f"Resource Audit: {audit}")
        return audit

    def run_xai(self, model, X: pd.DataFrame, y: pd.Series, name: str):
        """Global (Permutation) and Local (SHAP) XAI."""
        logger.info(f"Starting XAI analysis for {name}")
        
        # 1. Global: Permutation Importance (OBLIGATOIRE)
        if self.p8.enable_permutation_importance:
            logger.info("  - Computing Permutation Importance...")
            start_pi = time.time()
            r = permutation_importance(model, X, y, n_repeats=5, random_state=self.p8.random_state, n_jobs=-1)
            pi_df = pd.DataFrame({'feature': X.columns, 'importance': r.importances_mean})
            pi_df = pi_df.sort_values(by='importance', ascending=False)
            
            csv_path = self.output_dir / f"{name}_permutation_importance.csv"
            pi_df.to_csv(csv_path, index=False)
            logger.info(f"    * Global XAI saved to {csv_path} (Time: {time.time() - start_pi:.2f}s)")
        
        # 2. Local: SHAP (sur échantillon fixe)
        if self.p8.enable_shap:
            logger.info(f"  - Computing SHAP values (Sample size: {self.p8.xai_sample_rows})...")
            X_shap = X.head(self.p8.xai_sample_rows)
            try:
                # Heuristique pour le choix de l'explainer
                if hasattr(model, "feature_importances_") or "Forest" in str(type(model)):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_shap)
                else:
                    # Fallback Kernel SHAP (lent)
                    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 50))
                    shap_values = explainer.shap_values(X_shap)
                
                # Plotting
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_shap, show=False)
                plot_path = self.output_dir / f"{name}_shap_summary.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"    * Local XAI (SHAP) plot saved to {plot_path}")
            except Exception as e:
                logger.warning(f"    ! SHAP failed for {name}: {e}")
