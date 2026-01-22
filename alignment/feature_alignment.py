import numpy as np
import polars as pl
from typing import List, Dict, Tuple
from descriptors.statistics import ColumnDescriptors
from descriptors.distribution import DistributionComparator
from config.schema import AlignmentConfig

class FeatureAligner:
    """
    Aligne les caractéristiques entre CIC et TON en utilisant des mesures de similarité statistique.
    """
    def __init__(self, config: AlignmentConfig):
        self.config = config
        self.comparator = DistributionComparator()

    def compute_alignment_matrix(self, cic_df: pl.DataFrame, ton_df: pl.DataFrame) -> List[str]:
        """
        Identifie les colonnes communes basées sur le nom et valide leur similarité.
        """
        exclude = {"y", "source_file", "sample_id", "Label", "type", "ts"}
        cic_cols = [c for c in cic_df.columns if c not in exclude]
        ton_cols = [c for c in ton_df.columns if c not in exclude]
        
        # Intersection initiale par nom
        potential_common = sorted(list(set(cic_cols) & set(ton_cols)))
        
        final_common = []
        
        for col in potential_common:
            # Extraction des données pour test KS et Wasserstein
            # On limite à 1000 pour la performance de l'alignement
            s1 = cic_df[col].drop_nulls().head(1000).to_numpy()
            s2 = ton_df[col].drop_nulls().head(1000).to_numpy()
            
            if len(s1) < 10 or len(s2) < 10:
                continue
                
            # Calcul des descripteurs pour Cosine Similarity
            d1 = ColumnDescriptors.compute_all(cic_df[col])
            d2 = ColumnDescriptors.compute_all(ton_df[col])
            
            v1 = np.array(list(d1.values()))
            v2 = np.array(list(d2.values()))
            
            cos_sim = self.comparator.cosine_similarity(v1, v2)
            ks_p = self.comparator.ks_test(s1, s2)
            
            # Critères de sélection : Cosine >= seuil OU KS p-value >= seuil
            if cos_sim >= self.config.cosine_threshold or ks_p >= self.config.ks_threshold:
                final_common.append(col)
                
        return final_common
