#!/usr/bin/env python3
"""
Dataset loader for CIC-DDoS2019 and TON_IoT datasets
Handles downloading and loading of datasets for the IRP research project
Now with system monitoring, adaptive chunking, and incremental loading
"""
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Set, Callable
import warnings
from tqdm import tqdm
import gc  # Garbage collector for memory management
import pickle
from datetime import datetime

try:
    from system_monitor import SystemMonitor
except ImportError:
    SystemMonitor = None

warnings.filterwarnings('ignore')

# Setup logger
logger = logging.getLogger(__name__)

# Default chunk size (will be adjusted based on available RAM)
DEFAULT_CHUNK_SIZE = 5_000_000


class DatasetLoader:
    """Loader for CIC-DDoS2019 and TON_IoT datasets with system monitoring"""
    
    def __init__(self, data_dir: str = 'datasets', monitor: Optional[SystemMonitor] = None):
        """
        Initialize the dataset loader
        
        Args:
            data_dir: Root directory for datasets (default: 'datasets')
            monitor: Optional SystemMonitor instance for RAM/CPU monitoring
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Ensure subdirectories exist
        (self.data_dir / 'ton_iot').mkdir(parents=True, exist_ok=True)
        (self.data_dir / 'cic_ddos2019').mkdir(parents=True, exist_ok=True)
        
        # Initialize system monitor if not provided
        self.monitor = monitor
        if self.monitor is None and SystemMonitor is not None:
            try:
                self.monitor = SystemMonitor(max_memory_percent=90.0)
            except Exception as e:
                logger.warning(f"Could not initialize SystemMonitor: {e}. Continuing without monitoring.")
                self.monitor = None
        
        # Track loaded files for incremental loading
        self.loaded_files_cache_file = self.data_dir / '.loaded_files_cache.pkl'
        self.loaded_files: Set[str] = self._load_file_cache()
        
        # Callback for real-time updates (can be set from outside)
        self.progress_callback: Optional[Callable] = None
    
    def _load_file_cache(self) -> Set[str]:
        """Load cache of previously loaded files"""
        if self.loaded_files_cache_file.exists():
            try:
                with open(self.loaded_files_cache_file, 'rb') as f:
                    return set(pickle.load(f))
            except Exception as e:
                logger.warning(f"Could not load file cache: {e}")
        return set()
    
    def _save_file_cache(self):
        """Save cache of loaded files"""
        try:
            with open(self.loaded_files_cache_file, 'wb') as f:
                pickle.dump(list(self.loaded_files), f)
        except Exception as e:
            logger.warning(f"Could not save file cache: {e}")
    
    def _get_adaptive_chunk_size(self) -> int:
        """Get adaptive chunk size based on available RAM"""
        if self.monitor is None:
            return DEFAULT_CHUNK_SIZE
        
        try:
            return self.monitor.calculate_optimal_chunk_size(estimated_row_size_bytes=500)
        except Exception as e:
            logger.warning(f"Could not calculate adaptive chunk size: {e}. Using default.")
            return DEFAULT_CHUNK_SIZE
    
    def _analyze_csv_headers(self, file_path: Path, n_rows: int = 1000) -> Tuple[List[str], pd.DataFrame]:
        """
        Analyze CSV headers and a few rows to understand structure
        
        Args:
            file_path: Path to CSV file
            n_rows: Number of rows to read for analysis (default: 1000)
            
        Returns:
            Tuple of (column_names, sample_dataframe)
        """
        try:
            # Read only a small sample to analyze structure
            sample_df = pd.read_csv(file_path, nrows=n_rows, low_memory=False)
            columns = list(sample_df.columns)
            logger.debug(f"[ANALYSIS] Analyzed {len(columns)} columns from {file_path.name} (first {n_rows} rows)")
            return columns, sample_df
        except Exception as e:
            logger.warning(f"Could not analyze headers from {file_path}: {e}")
            # Fallback: read only first line to get column names
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    header_line = f.readline().strip()
                    columns = header_line.split(',')
                    # Clean column names
                    columns = [col.strip().strip('"').strip("'") for col in columns]
                    return columns, pd.DataFrame()
            except Exception as e2:
                logger.error(f"Could not read headers from {file_path}: {e2}")
                return [], pd.DataFrame()
    
    def _determine_common_features_before_load(self, ton_path: Optional[Path] = None, 
                                               cic_files: Optional[List[Path]] = None) -> Dict:
        """
        Determine common features BEFORE loading full datasets
        Analyzes only headers and a few rows to determine feature mapping
        
        Args:
            ton_path: Path to TON_IoT CSV file
            cic_files: List of paths to CIC-DDoS2019 CSV files (use first one for analysis)
            
        Returns:
            Dictionary with 'ton_columns', 'cic_columns', 'common_features', 'feature_mapping'
        """
        logger.info("[STEP] Détermination des features communes AVANT chargement complet...")
        logger.info("[ACTION] Analyse des headers et quelques lignes...")
        
        result = {
            'ton_columns': [],
            'cic_columns': [],
            'common_features': {},
            'sample_ton': None,
            'sample_cic': None
        }
        
        # Analyze TON_IoT
        if ton_path is None:
            possible_paths = [
                self.data_dir / 'ton_iot' / 'train_test_network.csv',
                self.data_dir / 'ton_iot' / 'windows10_dataset.csv',
                Path('train_test_network.csv'),
            ]
            for path in possible_paths:
                if path.exists():
                    ton_path = path
                    break
        
        if ton_path and ton_path.exists():
            ton_columns, sample_ton = self._analyze_csv_headers(ton_path, n_rows=1000)
            result['ton_columns'] = ton_columns
            result['sample_ton'] = sample_ton
            logger.info(f"[OUTPUT] TON_IoT: {len(ton_columns)} colonnes détectées")
        else:
            logger.warning("[WARNING] TON_IoT file not found for header analysis")
        
        # Analyze CIC-DDoS2019 (use first CSV file)
        if cic_files is None:
            cic_dir = self.data_dir / 'cic_ddos2019'
            cic_files = list(cic_dir.glob("*.csv"))
            # Also check subdirectories
            for subdir in cic_dir.iterdir():
                if subdir.is_dir():
                    cic_files.extend(list(subdir.glob("*.csv")))
                    for subdir2 in subdir.iterdir():
                        if subdir2.is_dir():
                            cic_files.extend(list(subdir2.glob("*.csv")))
            
            # Filter out examples
            cic_files = [f for f in cic_files if 'example' not in f.name.lower() 
                        and 'sample' not in f.name.lower() 
                        and 'template' not in f.name.lower()]
        
        if cic_files:
            # Use first CSV file for analysis
            cic_file = cic_files[0]
            cic_columns, sample_cic = self._analyze_csv_headers(cic_file, n_rows=1000)
            result['cic_columns'] = cic_columns
            result['sample_cic'] = sample_cic
            logger.info(f"[OUTPUT] CIC-DDoS2019: {len(cic_columns)} colonnes détectées (fichier: {cic_file.name})")
        else:
            logger.warning("[WARNING] CIC-DDoS2019 files not found for header analysis")
        
        # Use FeatureAnalyzer to determine common features
        if result['sample_ton'] is not None and result['sample_cic'] is not None:
            try:
                from feature_analyzer import FeatureAnalyzer
                analyzer = FeatureAnalyzer()
                common_features = analyzer.extract_common_features(result['sample_ton'], result['sample_cic'])
                result['common_features'] = common_features
                logger.info(f"[OUTPUT] {len(common_features)} features communes déterminées AVANT chargement complet")
            except Exception as e:
                logger.warning(f"[WARNING] FeatureAnalyzer failed: {e}, using basic matching")
                # Fallback: exact matches only
                common_exact = set(result['ton_columns']) & set(result['cic_columns'])
                result['common_features'] = [
                    {'unified_name': col, 'ton_name': col, 'cic_name': col, 'type': 'exact_match'}
                    for col in common_exact
                ]
        elif result['ton_columns'] and result['cic_columns']:
            # Basic matching if samples not available
            common_exact = set(result['ton_columns']) & set(result['cic_columns'])
            result['common_features'] = [
                {'unified_name': col, 'ton_name': col, 'cic_name': col, 'type': 'exact_match'}
                for col in common_exact
            ]
            logger.info(f"[OUTPUT] {len(common_exact)} features communes exactes trouvées")
        
        return result
    
    def _check_memory_and_gc(self, force: bool = False):
        """Check memory usage and trigger GC if needed"""
        if self.monitor is None:
            if force:
                gc.collect()
            return
        
        try:
            is_safe, msg = self.monitor.check_memory_safe()
            if not is_safe:
                logger.warning(f"[MEMORY] {msg}")
                gc.collect()
            elif force:
                gc.collect()
        except Exception as e:
            logger.debug(f"Memory check error: {e}")
            if force:
                gc.collect()
    
    def _sample_rows_efficient(self, total_rows: int, sample_ratio: float, random_state: int) -> np.ndarray:
        """
        Generate indices to sample efficiently without loading all data first
        
        Args:
            total_rows: Total number of rows in the file
            sample_ratio: Ratio to sample (0.1 = 10%)
            random_state: Random seed
            
        Returns:
            Array of row indices to keep (sorted for efficient reading)
        """
        if sample_ratio >= 1.0:
            return None  # Keep all rows
        
        np.random.seed(random_state)
        sample_size = max(1, int(total_rows * sample_ratio))
        indices = np.sort(np.random.choice(total_rows, size=sample_size, replace=False))
        return indices
    
    def _count_file_rows(self, file_path: Path) -> int:
        """Quickly count total rows in a CSV file (optimized for large files)"""
        try:
            # Fast method: count newlines in chunks to avoid loading entire file
            count = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    count += chunk.count(b'\n')
            # Adjust: file might not end with newline, and we subtract header
            return max(0, count - 1)
        except Exception as e:
            logger.warning(f"Could not count rows in {file_path.name}: {e}")
            return 0
    
    def load_ton_iot(self, file_path: Optional[str] = None, sample_ratio: float = 1.0, 
                     random_state: int = 42, incremental: bool = True) -> pd.DataFrame:
        """
        Load TON_IoT dataset with memory-efficient sampling
        
        Args:
            file_path: Path to TON_IoT CSV file. If None, looks for train_test_network.csv in root
            sample_ratio: Ratio of data to sample (1.0 = 100%, 0.1 = 10% for testing). Default: 1.0
            random_state: Random seed for sampling. Default: 42
            incremental: If True, skip if file already loaded. Default: True
            
        Returns:
            DataFrame containing TON_IoT data (sampled if sample_ratio < 1.0)
        """
        if file_path is None:
            # Try to find TON_IoT dataset in standard locations
            possible_paths = [
                self.data_dir / 'ton_iot' / 'train_test_network.csv',
                self.data_dir / 'ton_iot' / 'windows10_dataset.csv',
                Path('train_test_network.csv'),  # Legacy: root for backward compatibility
                Path('Processed_datasets/Processed_Windows_dataset/windows10_dataset.csv'),  # Legacy
                Path('data/raw/TON_IoT/train_test_network.csv'),  # Legacy
            ]
            
            for path in possible_paths:
                if path.exists():
                    file_path = str(path)
                    break
            
            if file_path is None:
                raise FileNotFoundError(
                    f"TON_IoT dataset not found. Please provide file_path or place "
                    f"train_test_network.csv in {self.data_dir / 'ton_iot'}/"
                )
        
        file_path_obj = Path(file_path)
        file_key = str(file_path_obj.resolve())
        
        # Check if already loaded (incremental mode)
        if incremental and file_key in self.loaded_files:
            logger.info(f"[STEP] TON_IoT file already loaded (incremental mode): {file_path_obj.name}")
            logger.info(f"[INFO] Use incremental=False to reload")
            return pd.DataFrame()  # Return empty, caller should handle cached data
        
        logger.info(f"[STEP] Loading TON_IoT dataset")
        logger.info(f"[INPUT] File path: {file_path}")
        
        # Determine common features BEFORE loading if possible
        common_features_info = None
        columns_to_load = None  # None = load all columns
        
        # Start progress tracking
        if self.monitor:
            self.monitor.start_progress_tracking()
        
        # Check file size
        if file_path_obj.exists():
            file_size_mb = file_path_obj.stat().st_size / (1024 * 1024)
            logger.info(f"[INFO] File size: {file_size_mb:.2f} MB")
        
        # Get adaptive chunk size
        chunk_size = self._get_adaptive_chunk_size()
        logger.info(f"[INFO] Using adaptive chunk size: {chunk_size:,} rows")
        
        try:
            # For very small sample ratios (< 0.01), use decimation approach (much faster)
            if sample_ratio < 1.0:
                if sample_ratio < 0.01:  # For very small ratios (< 1%), use decimation
                    logger.info(f"[ACTION] Using optimized decimation sampling for {sample_ratio*100:.2f}%...")
                    # Calculate decimation step (every Nth row)
                    decimation_step = int(1.0 / sample_ratio)
                    logger.info(f"[INFO] Reading every {decimation_step} row(s) (decimation step)")
                    
                    # Use pandas skiprows with function for efficient sampling
                    def skip_func(row_idx):
                        # Skip header (row 0) and sample based on decimation
                        return row_idx > 0 and (row_idx % decimation_step != 0)
                    
                    # Read with decimation (much faster than loading all then sampling)
                    df = pd.read_csv(file_path_obj, skiprows=skip_func, low_memory=False, 
                                   nrows=int(1e6))  # Limit to 1M rows max for safety
                    
                    # If we got more than needed due to decimation step approximation, sample down
                    target_size = max(100, int(len(df) * (sample_ratio / (1.0/decimation_step))))
                    if len(df) > target_size:
                        df = df.sample(n=target_size, random_state=random_state).reset_index(drop=True)
                    
                    logger.info(f"[OUTPUT] TON_IoT loaded with decimation: {df.shape[0]:,} rows, {df.shape[1]} columns")
                    return df
                else:
                    # For larger ratios (>= 1%), use chunk-based sampling
                    logger.info(f"[ACTION] Counting total rows for efficient sampling...")
                    total_rows = self._count_file_rows(file_path_obj)
                    if total_rows > 0:
                        sample_indices = self._sample_rows_efficient(total_rows, sample_ratio, random_state)
                        logger.info(f"[INFO] Will sample {len(sample_indices):,} rows from {total_rows:,} total")
            else:
                sample_indices = None
            
            # Load in chunks
            file_chunks = []
            chunk_count = 0
            rows_loaded = 0
            next_sample_idx = 0 if sample_indices is not None else None
            
            logger.info(f"[ACTION] Reading CSV file with chunk size: {chunk_size:,} rows")
            
            chunk_iterator = pd.read_csv(file_path, low_memory=False, chunksize=chunk_size)
            
            for chunk in chunk_iterator:
                chunk_start_row = chunk_count * chunk_size
                chunk_end_row = chunk_start_row + len(chunk)
                
                # If sampling, filter rows within this chunk
                if sample_indices is not None and next_sample_idx is not None:
                    # Find indices that fall within this chunk
                    chunk_sample_mask = (sample_indices >= chunk_start_row) & (sample_indices < chunk_end_row)
                    chunk_sample_indices = sample_indices[chunk_sample_mask]
                    
                    if len(chunk_sample_indices) > 0:
                        # Convert global indices to local chunk indices
                        local_indices = chunk_sample_indices - chunk_start_row
                        chunk = chunk.iloc[local_indices].reset_index(drop=True)
                        
                        logger.debug(f"[ACTION] Chunk {chunk_count + 1}: Kept {len(chunk):,} rows from chunk")
                    else:
                        # Skip this chunk entirely (no sampled rows in it)
                        logger.debug(f"[ACTION] Chunk {chunk_count + 1}: Skipped (no sampled rows)")
                        chunk_count += 1
                        gc.collect()
                        continue
                
                file_chunks.append(chunk)
                chunk_count += 1
                rows_loaded += len(chunk)
                
                # Memory check and GC
                if chunk_count % 5 == 0:
                    self._check_memory_and_gc(force=True)
                
                # Progress callback
                if self.progress_callback:
                    self.progress_callback({
                        'type': 'loading',
                        'dataset': 'TON_IoT',
                        'chunks_processed': chunk_count,
                        'rows_loaded': rows_loaded
                    })
            
            # Combine chunks
            if file_chunks:
                logger.info(f"[ACTION] Combining {len(file_chunks)} chunks...")
                df = pd.concat(file_chunks, ignore_index=True)
                del file_chunks
                self._check_memory_and_gc(force=True)
            else:
                logger.warning(f"[WARNING] TON_IoT file appears empty")
                df = pd.DataFrame()
            
            logger.info(f"[OUTPUT] TON_IoT loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
            if len(df) > 0:
                logger.info(f"[OUTPUT] Memory usage: ~{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Mark as loaded
            if incremental:
                self.loaded_files.add(file_key)
                self._save_file_cache()
            
            return df
            
        except FileNotFoundError as e:
            logger.error(f"[ERROR] TON_IoT file not found: {file_path}")
            logger.error(f"[ERROR] {e}", exc_info=True)
            raise
        except MemoryError as me:
            logger.error(f"[ERROR] Memory error loading TON_IoT: {me}")
            if self.monitor:
                mem_info = self.monitor.get_memory_info()
                logger.error(f"[ERROR] Current memory: {mem_info['used_percent']:.1f}%")
            logger.error(f"[ERROR] Try reducing sample_ratio or free up memory")
            raise
        except Exception as e:
            logger.error(f"[ERROR] Error loading TON_IoT dataset: {e}", exc_info=True)
            raise
    
    def load_cic_ddos2019(self, dataset_path: Optional[str] = None, sample_ratio: float = 1.0, 
                          random_state: int = 42, incremental: bool = True, 
                          detect_new_files: bool = True) -> pd.DataFrame:
        """
        Load CIC-DDoS2019 dataset with memory-efficient sampling and new file detection
        
        Args:
            dataset_path: Path to CIC-DDoS2019 directory. If None, looks in data/raw/CIC-DDoS2019/
            sample_ratio: Ratio of data to sample from each file (1.0 = 100%, 0.1 = 10% for testing). Default: 1.0
            random_state: Random seed for sampling. Default: 42
            incremental: If True, skip already loaded files. Default: True
            detect_new_files: If True, detect and load new CSV files. Default: True
            
        Returns:
            DataFrame containing CIC-DDoS2019 data (sampled if sample_ratio < 1.0)
        """
        if dataset_path is None:
            dataset_path = self.data_dir / 'cic_ddos2019'
            # Also check legacy location
            legacy_path = Path('data/raw/CIC-DDoS2019')
            if legacy_path.exists() and not dataset_path.exists():
                dataset_path = legacy_path
        
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"CIC-DDoS2019 dataset not found at {dataset_path}. "
                "Please download the dataset from: "
                "https://www.unb.ca/cic/datasets/ddos-2019.html "
                f"and place CSV files in {self.data_dir / 'cic_ddos2019'}/"
            )
        
        # Find all CSV files (recursively)
        all_csv_files = list(dataset_path.glob("*.csv"))
        subdir_csv_files = []
        
        for subdir in dataset_path.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('.'):
                # First level subdirectories
                subdir_csvs = list(subdir.glob("*.csv"))
                subdir_csv_files.extend(subdir_csvs)
                
                # Second level (e.g., examples/Training-Day01/)
                for subdir2 in subdir.iterdir():
                    if subdir2.is_dir() and not subdir2.name.startswith('.'):
                        subdir2_csvs = list(subdir2.glob("*.csv"))
                        subdir_csv_files.extend(subdir2_csvs)
        
        all_csv_files.extend(subdir_csv_files)
        
        # Filter out example/template files
        csv_files = [
            f for f in all_csv_files 
            if not any(excluded in f.name.lower() for excluded in ['example', 'sample', 'template', 'structure'])
        ]
        
        if not csv_files:
            raise FileNotFoundError(
                f"No valid CSV files found in {dataset_path} or its subdirectories. "
                f"Please download the CIC-DDoS2019 dataset."
            )
        
        # Filter to only new files if incremental mode
        if incremental and detect_new_files:
            new_files = [f for f in csv_files if str(f.resolve()) not in self.loaded_files]
            if new_files:
                logger.info(f"[INFO] Detected {len(new_files)} new CSV file(s) to load")
                csv_files = new_files
            elif len(self.loaded_files) > 0:
                logger.info(f"[INFO] All files already loaded. Use incremental=False to reload all.")
                return pd.DataFrame()
        
        logger.info(f"Found {len(csv_files)} CSV file(s) to process in CIC-DDoS2019 dataset")
        
        # Start progress tracking
        if self.monitor:
            self.monitor.start_progress_tracking()
        
        # Get adaptive chunk size
        chunk_size = self._get_adaptive_chunk_size()
        logger.info(f"[INFO] Using adaptive chunk size: {chunk_size:,} rows")
        
        all_chunks = []
        total_rows_loaded = 0
        
        # Progress bar with ETA
        progress_bar = tqdm(csv_files, desc="Loading CIC-DDoS2019 CSV files")
        
        for file_idx, csv_file in enumerate(progress_bar):
            try:
                logger.info(f"[ACTION] Loading file {file_idx + 1}/{len(csv_files)}: {csv_file.name}")
                
                # Update progress with ETA
                if self.monitor:
                    progress_info = self.monitor.update_progress(
                        file_idx, len(csv_files), item_name="files"
                    )
                    progress_bar.set_postfix({
                        'ETA': progress_info['eta_formatted'],
                        'RAM': f"{self.monitor.get_memory_info()['used_percent']:.1f}%"
                    })
                
                file_size_mb = csv_file.stat().st_size / (1024 * 1024)
                logger.debug(f"[INFO] File size: {file_size_mb:.2f} MB")
                
                # Optimized sampling for very small ratios
                sample_indices = None
                if sample_ratio < 1.0:
                    if sample_ratio < 0.01:  # For very small ratios (< 1%), use decimation
                        logger.info(f"[ACTION] Using optimized decimation sampling for {sample_ratio*100:.2f}%...")
                        decimation_step = int(1.0 / sample_ratio)
                        
                        def skip_func(row_idx):
                            return row_idx > 0 and (row_idx % decimation_step != 0)
                        
                        df_file = pd.read_csv(csv_file, skiprows=skip_func, low_memory=False, nrows=int(1e6))
                        
                        # Fine-tune sample size if needed
                        target_size = max(100, int(len(df_file) * (sample_ratio / (1.0/decimation_step))))
                        if len(df_file) > target_size:
                            df_file = df_file.sample(n=target_size, random_state=random_state).reset_index(drop=True)
                        
                        all_chunks.append(df_file)
                        logger.info(f"[OUTPUT] Loaded {csv_file.name}: {df_file.shape[0]:,} rows using decimation")
                        continue  # Skip chunk-based loading for this file
                    else:
                        # For larger ratios, use chunk-based approach
                        total_rows = self._count_file_rows(csv_file)
                        if total_rows > 0:
                            sample_indices = self._sample_rows_efficient(total_rows, sample_ratio, random_state)
                
                file_chunks = []
                chunk_count = 0
                next_sample_idx = 0 if sample_indices is not None else None
                
                try:
                    chunk_iterator = pd.read_csv(csv_file, low_memory=False, chunksize=chunk_size)
                    
                    for chunk in chunk_iterator:
                        chunk_start_row = chunk_count * chunk_size
                        chunk_end_row = chunk_start_row + len(chunk)
                        
                        # Filter sampled rows if needed
                        if sample_indices is not None and next_sample_idx is not None:
                            chunk_sample_mask = (sample_indices >= chunk_start_row) & (sample_indices < chunk_end_row)
                            chunk_sample_indices = sample_indices[chunk_sample_mask]
                            
                            if len(chunk_sample_indices) > 0:
                                local_indices = chunk_sample_indices - chunk_start_row
                                chunk = chunk.iloc[local_indices].reset_index(drop=True)
                            else:
                                chunk_count += 1
                                gc.collect()
                                continue
                        
                        file_chunks.append(chunk)
                        chunk_count += 1
                        total_rows_loaded += len(chunk)
                        
                        # Memory check
                        if chunk_count % 5 == 0:
                            self._check_memory_and_gc(force=True)
                        
                        # Progress callback
                        if self.progress_callback:
                            self.progress_callback({
                                'type': 'loading',
                                'dataset': 'CIC-DDoS2019',
                                'file': csv_file.name,
                                'file_idx': file_idx + 1,
                                'total_files': len(csv_files),
                                'chunks_processed': chunk_count,
                                'rows_loaded': total_rows_loaded
                            })
                    
                    # Combine chunks from this file
                    if file_chunks:
                        df_file = pd.concat(file_chunks, ignore_index=True)
                        all_chunks.append(df_file)
                        logger.info(f"[OUTPUT] Loaded {csv_file.name}: {df_file.shape[0]:,} rows, {df_file.shape[1]} columns")
                        
                        del file_chunks, df_file
                        self._check_memory_and_gc(force=True)
                        
                        # Mark file as loaded
                        if incremental:
                            self.loaded_files.add(str(csv_file.resolve()))
                    
                except pd.errors.EmptyDataError:
                    logger.warning(f"[WARNING] File {csv_file.name} is empty or corrupted")
                    continue
                except MemoryError as me:
                    logger.error(f"[ERROR] Memory error loading {csv_file.name}: {me}")
                    if self.monitor:
                        mem_info = self.monitor.get_memory_info()
                        logger.error(f"[ERROR] Current memory: {mem_info['used_percent']:.1f}%")
                    raise
                    
            except Exception as e:
                logger.error(f"[ERROR] Could not load {csv_file.name}: {e}", exc_info=True)
                logger.warning(f"[WARNING] Skipping file {csv_file.name} due to error")
                continue
        
        # Save file cache
        if incremental:
            self._save_file_cache()
        
        if not all_chunks:
            error_msg = "[ERROR] No valid CSV files could be loaded from CIC-DDoS2019"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Concatenate all dataframes
        logger.info(f"[STEP] Concatenating {len(all_chunks)} loaded dataframes")
        logger.info(f"[INPUT] Total rows accumulated: {total_rows_loaded:,}")
        
        try:
            combined_df = pd.concat(all_chunks, ignore_index=True)
            
            del all_chunks
            self._check_memory_and_gc(force=True)
            
            logger.info(f"[OUTPUT] CIC-DDoS2019 loaded successfully: {combined_df.shape[0]:,} rows, {combined_df.shape[1]} columns")
            if len(combined_df) > 0:
                logger.info(f"[OUTPUT] Memory usage: ~{combined_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            return combined_df
            
        except MemoryError as me:
            logger.error(f"[ERROR] Memory error during concatenation: {me}")
            if self.monitor:
                mem_info = self.monitor.get_memory_info()
                logger.error(f"[ERROR] Current memory: {mem_info['used_percent']:.1f}%")
            raise
        except Exception as e:
            logger.error(f"[ERROR] Error concatenating dataframes: {e}", exc_info=True)
            raise
    
    def get_attack_types(self, df: pd.DataFrame, label_col: Optional[str] = None) -> List[str]:
        """Get unique attack types from CIC-DDoS2019 dataset"""
        if label_col is None:
            label_candidates = ['Label', 'label', 'Attack', 'attack', 'Class', 'class']
            for col in label_candidates:
                if col in df.columns:
                    label_col = col
                    break
        
        if label_col is None or label_col not in df.columns:
            return []
        
        attack_types = df[label_col].unique().tolist()
        return sorted([str(at) for at in attack_types])
    
    def get_dataset_info(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """Get information about a dataset"""
        info = {
            'name': dataset_name,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Check for label column
        label_candidates = ['Label', 'label', 'Attack', 'attack', 'Class', 'class']
        label_col = None
        for col in label_candidates:
            if col in df.columns:
                label_col = col
                info['label_column'] = col
                info['unique_labels'] = df[col].value_counts().to_dict()
                break
        
        # CIC-DDoS2019 specific validations
        if 'CIC-DDoS2019' in dataset_name or 'cic' in dataset_name.lower():
            feature_count = len([c for c in df.columns if c != label_col])
            info['feature_count'] = feature_count
            info['expected_features'] = 80
            
            if abs(feature_count - 80) > 10:
                import warnings
                warnings.warn(
                    f"CIC-DDoS2019 dataset has {feature_count} features, expected ~80.",
                    UserWarning
                )
            
            if label_col:
                attack_types = self.get_attack_types(df, label_col)
                info['attack_types'] = attack_types
                info['attack_type_count'] = len(attack_types)
        
        return info


def main():
    """Test the dataset loader"""
    loader = DatasetLoader()
    
    # Test TON_IoT loading
    try:
        ton_iot = loader.load_ton_iot()
        ton_info = loader.get_dataset_info(ton_iot, "TON_IoT")
        print("\nTON_IoT Info:")
        print(f"  Shape: {ton_info['shape']}")
        print(f"  Label column: {ton_info.get('label_column', 'Not found')}")
        print(f"  Memory usage: {ton_info['memory_usage_mb']:.2f} MB")
    except Exception as e:
        print(f"Warning: Could not load TON_IoT: {e}")
    
    # Test CIC-DDoS2019 loading
    try:
        cic_ddos = loader.load_cic_ddos2019()
        cic_info = loader.get_dataset_info(cic_ddos, "CIC-DDoS2019")
        print("\nCIC-DDoS2019 Info:")
        print(f"  Shape: {cic_info['shape']}")
        print(f"  Label column: {cic_info.get('label_column', 'Not found')}")
        print(f"  Memory usage: {cic_info['memory_usage_mb']:.2f} MB")
    except Exception as e:
        print(f"\nCIC-DDoS2019 not available: {e}")


if __name__ == "__main__":
    main()
