import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import recall_score, f1_score, roc_auc_score

from src.new_pipeline.config import config

logger = logging.getLogger(__name__)

class LateFusionManager:
    """Phase 5: Late Fusion Optimization"""

    def __init__(self):
        self.p5 = config.phase5
        self.label_col = config.phase0.label_col

    def optimize_fusion(self, cic_val_path: Path, ton_val_path: Path, 
                        m_cic_path: Path, m_ton_path: Path, features: List[str]) -> float:
        """Finds the best weight w for p_final = w * p_cic + (1-w) * p_ton."""
        logger.info("Starting Phase 5: Late Fusion Optimization")
        
        m_cic = joblib.load(m_cic_path)
        m_ton = joblib.load(m_ton_path)
        
        df_cic = pd.read_parquet(cic_val_path)
        df_ton = pd.read_parquet(ton_val_path)
        
        # We need to evaluate both models on the SAME validation set to optimize fusion
        # Or more accurately, we evaluate the fusion on a combined validation set
        X_cic = df_cic[features]
        y_cic = df_cic[self.label_col]
        
        X_ton = df_ton[features]
        y_ton = df_ton[self.label_col]
        
        # Combined validation for weight optimization
        X_comb = pd.concat([X_cic, X_ton])
        y_comb = pd.concat([y_cic, y_ton])
        
        p_cic = m_cic.predict_proba(X_comb)[:, 1]
        p_ton = m_ton.predict_proba(X_comb)[:, 1]
        
        weights = np.arange(0, 1.0001, self.p5.weight_grid_step)
        best_w = 0.5
        best_score = -1
        results = []

        for w in weights:
            p_final = w * p_cic + (1 - w) * p_ton
            y_pred = (p_final >= self.p5.threshold).astype(int)
            
            if self.p5.optimize_metric == "recall":
                score = recall_score(y_comb, y_pred)
            elif self.p5.optimize_metric == "f1":
                score = f1_score(y_comb, y_pred)
            else:
                score = roc_auc_score(y_comb, p_final)
                
            results.append({'w': w, 'score': score})
            if score > best_score:
                best_score = score
                best_w = w

        # Save results
        self.p5.fusion_artifacts_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(self.p5.fusion_artifacts_dir / self.p5.fusion_curve_csv, index=False)
        
        fusion_config = {
            'best_w': float(best_w),
            'best_score': float(best_score),
            'metric': self.p5.optimize_metric,
            'threshold': self.p5.threshold
        }
        with open(self.p5.fusion_artifacts_dir / self.p5.fusion_config_json, 'w') as f:
            json.dump(fusion_config, f)
            
        logger.info(f"Phase 5 completed. Best weight w={best_w:.2f} (Score: {best_score:.4f})")
        return best_w

    def predict_fusion(self, X: pd.DataFrame, m_cic, m_ton, w: float, threshold: float = 0.5) -> np.ndarray:
        """Performs late fusion prediction."""
        p_cic = m_cic.predict_proba(X)[:, 1]
        p_ton = m_ton.predict_proba(X)[:, 1]
        p_final = w * p_cic + (1 - w) * p_ton
        return (p_final >= threshold).astype(int), p_final
