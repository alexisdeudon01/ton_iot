import logging
import json
import time
from pathlib import Path
from typing import List, Set, Dict

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, f_classif

from src.new_pipeline.config import config

logger = logging.getLogger(__name__)

class FeatureSelector:
    """Phase 3: Feature Selection (Par Dataset) avec Mutual Info et F-Classif."""

    def __init__(self):
        self.p3 = config.phase3
        self.label_col = config.phase0.label_col
        logger.info(f"Initialized FeatureSelector (Top-K: {self.p3.top_k})")

    def select_features(self, cic_train_path: Path, ton_train_path: Path, f_common: List[str]) -> List[str]:
        """
        Select top features from both datasets using a hybrid approach and intersect.
        CORRECTION: Tri déterministe de f_fusion pour la reproductibilité.
        """
        logger.info("Starting Phase 3: Hybrid Feature Selection")
        start_time = time.time()
        
        df_cic = pd.read_parquet(cic_train_path)
        df_ton = pd.read_parquet(ton_train_path)
        
        def get_hybrid_top_k(df, name):
            logger.info(f"  - Analyzing features for {name}...")
            X = df[f_common]
            y = df[self.label_col]
            
            # 1. Mutual Information
            logger.info(f"    * Computing Mutual Information...")
            mi = mutual_info_classif(X, y, random_state=config.phase1.random_state)
            mi_norm = (mi - mi.min()) / (mi.max() - mi.min() + 1e-9)
            
            # 2. F-Classif (ANOVA)
            logger.info(f"    * Computing F-Classif scores...")
            f_scores, _ = f_classif(X, y)
            f_scores = np.nan_to_num(f_scores)
            f_norm = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-9)
            
            # Hybrid Score (Mean of normalized scores)
            hybrid_score = (mi_norm + f_norm) / 2
            scores_df = pd.DataFrame({
                'feature': f_common,
                'mi': mi,
                'f_score': f_scores,
                'hybrid': hybrid_score
            }).sort_values(by='hybrid', ascending=False)
            
            top_features = set(scores_df.head(self.p3.top_k)['feature'])
            logger.info(f"    * Top 5 features for {name}: {list(top_features)[:5]}")
            return top_features

        top_cic = get_hybrid_top_k(df_cic, "CICDDoS2019")
        top_ton = get_hybrid_top_k(df_ton, "ToN-IoT")
        
        # Intersection et Tri Déterministe (CORRECTION)
        f_fusion = sorted(list(top_cic.intersection(top_ton)))
        logger.info(f"Intersection found {len(f_fusion)} common top features.")
        
        if len(f_fusion) < self.p3.min_fusion_features:
            logger.warning(f"Intersection too small ({len(f_fusion)} < {self.p3.min_fusion_features}). Falling back to F_common.")
            f_fusion = sorted(f_common)
        
        # Save F_fusion artifact
        self.p3.selection_artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = self.p3.selection_artifacts_dir / self.p3.f_fusion_json
        with open(artifact_path, 'w') as f:
            json.dump(f_fusion, f, indent=4)
            
        logger.info(f"Phase 3 completed in {time.time() - start_time:.2f}s. Artifact: {artifact_path}")
        return f_fusion
