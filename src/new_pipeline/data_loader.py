import logging
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import dask.dataframe as dd
from scipy import stats
from scipy.stats import wasserstein_distance, entropy
from sklearn.metrics.pairwise import cosine_similarity

from src.new_pipeline.config import config
from src.core.exceptions import DataLoadingError

logger = logging.getLogger(__name__)

class LateFusionDataLoader:
    """
    Phase 0: Data Cleaning & Consolidation (CSV -> Parquet)
    Phase 1: Feature Alignment (F_common)
    """

    def __init__(self):
        self.paths = config.paths
        self.p0 = config.phase0
        self.p1 = config.phase1
        
        # Ensure directories exist
        self.paths.data_parquet_root.mkdir(parents=True, exist_ok=True)
        self.p1.alignment_artifacts_dir.mkdir(parents=True, exist_ok=True)

    def prepare_datasets(self, force: bool = False):
        """Phase 0: Convert CSVs to Parquet with cleaning and harmonization."""
        logger.info("Starting Phase 0: Data Cleaning & Consolidation")
        
        if not force and self.paths.cic_parquet.exists() and self.paths.ton_parquet.exists():
            logger.info("Parquet datasets already exist. Skipping conversion.")
            return

        # Clean previous attempts
        import shutil
        if self.paths.cic_parquet.exists(): shutil.rmtree(self.paths.cic_parquet)
        if self.paths.ton_parquet.exists(): shutil.rmtree(self.paths.ton_parquet)

        self._prepare_cic()
        self._prepare_ton()
        logger.info("Phase 0 completed successfully.")

    def _prepare_cic(self):
        """Consolidate CICDDoS2019 CSVs into Parquet with strict schema enforcement."""
        logger.info(f"Consolidating CICDDoS2019 from {self.paths.cic_dir}")
        csv_files = list(self.paths.cic_dir.rglob("*.csv"))
        if not csv_files:
            raise DataLoadingError(f"No CSV files found in {self.paths.cic_dir}")

        rename_map = {
            'Source IP': 'src_ip', 'Destination IP': 'dst_ip',
            'Source Port': 'src_port', 'Destination Port': 'dst_port',
            'Protocol': 'proto', 'Timestamp': 'timestamp'
        }

        first_chunk = True
        common_schema = None

        for csv_path in csv_files:
            logger.info(f"Processing {csv_path.name}...")
            try:
                # Read first few rows to get columns
                sample = pd.read_csv(csv_path, nrows=1)
                sample.columns = [c.strip() for c in sample.columns]
                
                chunks = pd.read_csv(csv_path, chunksize=self.p0.chunksize, low_memory=self.p0.low_memory)
                for i, chunk in enumerate(chunks):
                    chunk.columns = [c.strip() for c in chunk.columns]
                    chunk.rename(columns=rename_map, inplace=True)
                    
                    # Label mapping
                    if self.p0.cic_label_col in chunk.columns:
                        chunk[self.p0.label_col] = (chunk[self.p0.cic_label_col].astype(str).str.strip().str.upper() != self.p0.cic_benign_value).astype(int)
                        chunk.drop(columns=[self.p0.cic_label_col], inplace=True)
                    
                    if self.p0.cic_add_source_col:
                        chunk[self.p0.cic_source_col] = str(csv_path)

                    for col in self.p0.drop_columns_if_present:
                        if col in chunk.columns: chunk.drop(columns=[col], inplace=True)

                    # Force consistent dtypes and handle Inf/NaN
                    for col in chunk.columns:
                        if col in ['src_ip', 'dst_ip', 'proto', 'timestamp', 'Flow ID', 'SimillarHTTP', 'source']:
                            chunk[col] = chunk[col].astype(str).fillna("N/A")
                        elif col == self.p0.label_col:
                            chunk[col] = chunk[col].astype('int64')
                        else:
                            chunk[col] = pd.to_numeric(chunk[col], errors='coerce').replace([np.inf, -np.inf], np.nan).astype('float64').fillna(0.0)

                    # Strict schema enforcement for Dask append
                    if common_schema is None:
                        common_schema = chunk.dtypes.to_dict()
                        cols_order = chunk.columns.tolist()
                    else:
                        # Add missing columns
                        for col, dtype in common_schema.items():
                            if col not in chunk.columns:
                                chunk[col] = "N/A" if dtype == object else 0.0
                                chunk[col] = chunk[col].astype(dtype)
                        # Drop extra columns not in first chunk
                        chunk = chunk[cols_order].astype(common_schema)

                    target_path = self.paths.cic_parquet
                    target_path.mkdir(parents=True, exist_ok=True)
                    dd_chunk = dd.from_pandas(chunk, npartitions=1)
                    dd_chunk.to_parquet(target_path, engine="pyarrow", append=not first_chunk, ignore_divisions=True)
                    first_chunk = False
            except Exception as e:
                logger.error(f"Error processing {csv_path}: {e}")

    def _prepare_ton(self):
        """Filter and convert ToN-IoT CSV to Parquet with strict schema enforcement."""
        logger.info(f"Processing ToN-IoT from {self.paths.ton_file}")
        if not self.paths.ton_file.exists():
            raise DataLoadingError(f"ToN-IoT file not found: {self.paths.ton_file}")

        rename_map = {'ts': 'timestamp'}
        first_chunk = True
        common_schema = None
        cols_order = None
        
        chunks = pd.read_csv(self.paths.ton_file, chunksize=self.p0.chunksize, low_memory=self.p0.low_memory)

        for i, chunk in enumerate(chunks):
            chunk.columns = [c.strip() for c in chunk.columns]
            chunk.rename(columns=rename_map, inplace=True)
            
            if self.p0.ton_type_col in chunk.columns:
                chunk = chunk[chunk[self.p0.ton_type_col].str.strip().str.lower().isin(self.p0.ton_allowed_types)]
                chunk[self.p0.label_col] = (chunk[self.p0.ton_type_col].str.strip().str.lower() == "ddos").astype(int)
                chunk.drop(columns=[self.p0.ton_type_col], inplace=True)
            
            for col in self.p0.drop_columns_if_present:
                if col in chunk.columns: chunk.drop(columns=[col], inplace=True)

            # Force consistent dtypes
            for col in chunk.columns:
                if col in ['src_ip', 'dst_ip', 'proto', 'timestamp', 'source']:
                    chunk[col] = chunk[col].astype(str).fillna("N/A")
                elif col == self.p0.label_col:
                    chunk[col] = chunk[col].astype('int64')
                else:
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce').replace([np.inf, -np.inf], np.nan).astype('float64').fillna(0.0)

            if common_schema is None:
                common_schema = chunk.dtypes.to_dict()
                cols_order = chunk.columns.tolist()
            else:
                for col, dtype in common_schema.items():
                    if col not in chunk.columns:
                        chunk[col] = "N/A" if dtype == object else 0.0
                        chunk[col] = chunk[col].astype(dtype)
                chunk = chunk[cols_order].astype(common_schema)

            target_path = self.paths.ton_parquet
            target_path.mkdir(parents=True, exist_ok=True)
            dd_chunk = dd.from_pandas(chunk, npartitions=1)
            dd_chunk.to_parquet(target_path, engine="pyarrow", append=not first_chunk, ignore_divisions=True)
            first_chunk = False

    def align_features(self) -> List[str]:
        """Phase 1: Feature Alignment (F_common) with advanced statistics."""
        logger.info("Starting Phase 1: Feature Alignment")
        
        def load_robust_sample(path, n_rows):
            ddf = dd.read_parquet(path)
            try:
                # Sample more to ensure we have enough after head
                df = ddf.sample(frac=min(1.0, (n_rows*10)/len(ddf)), random_state=42).compute()
                return df.head(n_rows)
            except:
                return ddf.head(n_rows)

        df_cic = load_robust_sample(self.paths.cic_parquet, self.p1.align_sample_rows)
        df_ton = load_robust_sample(self.paths.ton_parquet, self.p1.align_sample_rows)

        cic_num = df_cic.select_dtypes(include=[np.number]).drop(columns=[self.p0.label_col], errors='ignore')
        ton_num = df_ton.select_dtypes(include=[np.number]).drop(columns=[self.p0.label_col], errors='ignore')

        def get_descriptors(df):
            desc = {}
            for col in df.columns:
                data = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
                if data.isna().all(): continue
                median_val = data.median()
                data = data.fillna(median_val if not np.isnan(median_val) else 0)
                counts = data.value_counts()
                ent = entropy(counts)
                desc[col] = {
                    'mean': float(data.mean()), 'std': float(data.std()), 'median': float(data.median()),
                    'skew': float(data.skew()), 'kurtosis': float(data.kurtosis()), 'entropy': float(ent),
                    'zeros': float((data == 0).mean()), 'min': float(data.min()), 'max': float(data.max())
                }
            return pd.DataFrame(desc).T.fillna(0)

        logger.info("Computing statistical descriptors...")
        desc_cic = get_descriptors(cic_num)
        desc_ton = get_descriptors(ton_num)

        mapping = []
        f_common = []
        cic_cols = [c for c in cic_num.columns if c in desc_cic.index]
        ton_cols = [t for t in ton_num.columns if t in desc_ton.index]

        for c_col in cic_cols:
            best_match = None
            max_sim = -1
            for t_col in ton_cols:
                sim = cosine_similarity(desc_cic.loc[c_col].values.reshape(1, -1), desc_ton.loc[t_col].values.reshape(1, -1))[0][0]
                if sim > max_sim:
                    max_sim = sim
                    best_match = t_col
            
            if max_sim > 0.5:
                v_cic = cic_num[c_col].fillna(0)
                v_ton = ton_num[best_match].fillna(0)
                ks_stat, p_val = stats.ks_2samp(v_cic, v_ton)
                w_dist = wasserstein_distance(v_cic, v_ton)
                mapping.append({'cic_feature': c_col, 'ton_feature': best_match, 'cosine_sim': max_sim, 'ks_pvalue': p_val, 'wasserstein': w_dist})
                if max_sim > self.p1.descriptor_sim_threshold or p_val > self.p1.ks_pvalue_threshold:
                    f_common.append(c_col)

        mapping_df = pd.DataFrame(mapping)
        mapping_df.to_csv(self.p1.alignment_artifacts_dir / self.p1.mapping_pairs_csv, index=False)
        with open(self.p1.alignment_artifacts_dir / self.p1.f_common_json, 'w') as f:
            json.dump(sorted(f_common), f) # Sorted for determinism
            
        logger.info(f"Phase 1 completed. Found {len(f_common)} common features.")
        return sorted(f_common)
