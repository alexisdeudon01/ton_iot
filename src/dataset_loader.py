#!/usr/bin/env python3
"""
Dataset loader for CIC-DDoS2019 and TON_IoT datasets
Handles downloading and loading of datasets for the IRP research project
"""
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import warnings
from tqdm import tqdm
import gc  # Garbage collector for memory management

warnings.filterwarnings('ignore')

# Setup logger
logger = logging.getLogger(__name__)

# Chunk size for CSV loading (5 million rows)
CHUNK_SIZE = 5_000_000


class DatasetLoader:
    """Loader for CIC-DDoS2019 and TON_IoT datasets"""
    
    def __init__(self, data_dir: str = 'datasets'):
        """
        Initialize the dataset loader
        
        Args:
            data_dir: Root directory for datasets (default: 'datasets')
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Ensure subdirectories exist
        (self.data_dir / 'ton_iot').mkdir(parents=True, exist_ok=True)
        (self.data_dir / 'cic_ddos2019').mkdir(parents=True, exist_ok=True)
        
    def load_ton_iot(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load TON_IoT dataset
        
        Args:
            file_path: Path to TON_IoT CSV file. If None, looks for train_test_network.csv in root
            
        Returns:
            DataFrame containing TON_IoT data
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
        
        logger.info(f"[STEP] Loading TON_IoT dataset")
        logger.info(f"[INPUT] File path: {file_path}")
        
        # Check file size
        file_path_obj = Path(file_path)
        if file_path_obj.exists():
            file_size_mb = file_path_obj.stat().st_size / (1024 * 1024)
            logger.info(f"[INFO] File size: {file_size_mb:.2f} MB")
        
        try:
            # Load in chunks if file is large
            file_chunks = []
            chunk_count = 0
            
            logger.info(f"[ACTION] Reading CSV file with chunk size: {CHUNK_SIZE:,} rows")
            
            for chunk in pd.read_csv(file_path, low_memory=False, chunksize=CHUNK_SIZE):
                logger.debug(f"[ACTION] Processing chunk {chunk_count + 1}: {chunk.shape}")
                file_chunks.append(chunk)
                chunk_count += 1
                
                # Garbage collection every 5 chunks
                if chunk_count % 5 == 0:
                    logger.debug(f"[ACTION] Garbage collection triggered")
                    gc.collect()
            
            # Combine chunks
            if file_chunks:
                df = pd.concat(file_chunks, ignore_index=True)
                del file_chunks
                gc.collect()
            else:
                logger.warning(f"[WARNING] TON_IoT file appears empty")
                df = pd.DataFrame()
            
            logger.info(f"[OUTPUT] TON_IoT loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
            logger.info(f"[OUTPUT] Memory usage: ~{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            return df
            
        except FileNotFoundError as e:
            logger.error(f"[ERROR] TON_IoT file not found: {file_path}")
            logger.error(f"[ERROR] {e}", exc_info=True)
            raise
        except MemoryError as me:
            logger.error(f"[ERROR] Memory error loading TON_IoT: {me}")
            logger.error(f"[ERROR] Try reducing CHUNK_SIZE or free up memory")
            raise
        except Exception as e:
            logger.error(f"[ERROR] Error loading TON_IoT dataset: {e}", exc_info=True)
            raise
    
    def load_cic_ddos2019(self, dataset_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load CIC-DDoS2019 dataset
        
        The CIC-DDoS2019 dataset is a comprehensive DDoS attack dataset containing:
        - 80 network traffic features extracted using CICFlowMeter software
        - 11 types of DDoS attacks (reflective DDoS: DNS, LDAP, MSSQL, TFTP; UDP, UDP-Lag, SYN, etc.)
        - Both benign and attack traffic flows
        
        Reference: "Developing Realistic Distributed Denial of Service (DDoS) Attack 
        Dataset and Taxonomy" by Sharafaldin et al. (2019), Canadian Institute for Cybersecurity (CIC).
        
        CICFlowMeter is publicly available software from CIC used to extract network flow features.
        The dataset contains approximately 80 features per flow, covering various aspects of 
        network traffic behavior.
        
        Args:
            dataset_path: Path to CIC-DDoS2019 directory or CSV file. If None, looks in data/raw/CIC-DDoS2019/
            
        Returns:
            DataFrame containing CIC-DDoS2019 data with 80 CICFlowMeter features
            
        Raises:
            FileNotFoundError: If dataset directory or CSV files not found
            ValueError: If no valid CSV files could be loaded
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
        
        # CIC-DDoS2019 typically contains multiple CSV files (one per attack type)
        # Exclude example files and documentation files
        all_csv_files = list(dataset_path.glob("*.csv"))
        
        # Also check subdirectories (Training-Day01, Test-Day02, etc.)
        # Allow loading from examples/Training-Day01/ and examples/Test-Day02/ 
        # but exclude files with "example", "sample", "template", "structure" in their name
        subdir_csv_files = []
        for subdir in dataset_path.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('.'):
                # Check first level subdirectories
                subdir_csvs = list(subdir.glob("*.csv"))
                subdir_csv_files.extend(subdir_csvs)
                logger.debug(f"Found {len(subdir_csvs)} CSV files in subdirectory {subdir.name}/")
                
                # Also check second level (e.g., examples/Training-Day01/)
                for subdir2 in subdir.iterdir():
                    if subdir2.is_dir() and not subdir2.name.startswith('.'):
                        subdir2_csvs = list(subdir2.glob("*.csv"))
                        subdir_csv_files.extend(subdir2_csvs)
                        logger.debug(f"Found {len(subdir2_csvs)} CSV files in subdirectory {subdir.name}/{subdir2.name}/")
        
        # Combine all CSV files
        all_csv_files.extend(subdir_csv_files)
        
        # Filter out example/template files (by filename pattern only, not by directory)
        # This allows loading real data from examples/Training-Day01/ while excluding example_*.csv files
        csv_files = [
            f for f in all_csv_files 
            if not any(excluded in f.name.lower() for excluded in ['example', 'sample', 'template', 'structure'])
        ]
        
        if not csv_files:
            # If only example files were found, provide a helpful error message
            if all_csv_files:
                example_files = [f.name for f in all_csv_files[:5]]  # Show first 5
                if len(all_csv_files) > 5:
                    example_files.append(f"... and {len(all_csv_files) - 5} more")
                raise FileNotFoundError(
                    f"Only example/template CSV files found in {dataset_path} (and subdirectories): {example_files}. "
                    f"Please download the actual CIC-DDoS2019 dataset from "
                    f"https://www.unb.ca/cic/datasets/ddos-2019.html and place the CSV files in {dataset_path}/ "
                    f"or in subdirectories like Training-Day01/ or Test-Day02/. "
                    f"Example files are excluded from loading to prevent data corruption."
                )
            raise FileNotFoundError(
                f"No CSV files found in {dataset_path} or its subdirectories. "
                f"Please download the CIC-DDoS2019 dataset from "
                f"https://www.unb.ca/cic/datasets/ddos-2019.html and place CSV files in {dataset_path}/"
            )
        
        if len(all_csv_files) > len(csv_files):
            excluded_count = len(all_csv_files) - len(csv_files)
            logger.info(f"Excluded {excluded_count} example/template file(s) from loading")
        
        logger.info(f"Found {len(csv_files)} CSV files in CIC-DDoS2019 dataset "
                   f"(checked {dataset_path} and subdirectories)")
        
        # Load and concatenate all CSV files with progress bar using chunks for large files
        logger.info(f"[STEP] Starting CSV file loading with chunk size: {CHUNK_SIZE:,} rows")
        logger.info(f"[INPUT] {len(csv_files)} CSV files to process")
        
        all_chunks = []
        total_rows_loaded = 0
        
        for csv_file in tqdm(csv_files, desc="Loading CIC-DDoS2019 CSV files"):
            try:
                logger.info(f"[ACTION] Loading file: {csv_file.name}")
                logger.debug(f"[INPUT] File path: {csv_file.absolute()}")
                
                # Check file size first
                file_size_mb = csv_file.stat().st_size / (1024 * 1024)
                logger.debug(f"[INFO] File size: {file_size_mb:.2f} MB")
                
                # Load file in chunks if it's potentially large
                file_chunks = []
                chunk_count = 0
                
                try:
                    for chunk in pd.read_csv(csv_file, low_memory=False, chunksize=CHUNK_SIZE):
                        logger.debug(f"[ACTION] Processing chunk {chunk_count + 1} from {csv_file.name}: {chunk.shape}")
                        file_chunks.append(chunk)
                        chunk_count += 1
                        total_rows_loaded += len(chunk)
                        
                        # Force garbage collection after each chunk to free memory
                        if chunk_count % 5 == 0:
                            logger.debug(f"[ACTION] Garbage collection triggered after {chunk_count} chunks")
                            gc.collect()
                    
                    # Combine chunks from this file
                    if file_chunks:
                        df_file = pd.concat(file_chunks, ignore_index=True)
                        all_chunks.append(df_file)
                        logger.info(f"[OUTPUT] Loaded {csv_file.name}: {df_file.shape[0]:,} rows, {df_file.shape[1]} columns ({chunk_count} chunk(s))")
                        
                        # Clear intermediate data
                        del file_chunks, df_file
                        gc.collect()
                    else:
                        logger.warning(f"[WARNING] No data loaded from {csv_file.name} (file may be empty)")
                        
                except pd.errors.EmptyDataError:
                    logger.warning(f"[WARNING] File {csv_file.name} is empty or corrupted")
                    continue
                except MemoryError as me:
                    logger.error(f"[ERROR] Memory error loading {csv_file.name}: {me}")
                    logger.error(f"[ERROR] Try reducing CHUNK_SIZE or free up memory")
                    raise
                    
            except Exception as e:
                logger.error(f"[ERROR] Could not load {csv_file.name}: {e}", exc_info=True)
                logger.warning(f"[WARNING] Skipping file {csv_file.name} due to error")
                continue
        
        if not all_chunks:
            error_msg = "[ERROR] No valid CSV files could be loaded from CIC-DDoS2019"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Concatenate all dataframes
        logger.info(f"[STEP] Concatenating {len(all_chunks)} loaded dataframes")
        logger.info(f"[INPUT] Total rows accumulated: {total_rows_loaded:,}")
        logger.info(f"[ACTION] Merging dataframes...")
        
        try:
            combined_df = pd.concat(all_chunks, ignore_index=True)
            
            # Final garbage collection
            del all_chunks
            gc.collect()
            
            logger.info(f"[OUTPUT] CIC-DDoS2019 loaded successfully: {combined_df.shape[0]:,} rows, {combined_df.shape[1]} columns")
            logger.info(f"[OUTPUT] Memory usage: ~{combined_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            return combined_df
            
        except MemoryError as me:
            logger.error(f"[ERROR] Memory error during concatenation: {me}")
            logger.error(f"[ERROR] Try processing files separately or reduce dataset size")
            raise
        except Exception as e:
            logger.error(f"[ERROR] Error concatenating dataframes: {e}", exc_info=True)
            raise
    
    def get_attack_types(self, df: pd.DataFrame, label_col: Optional[str] = None) -> List[str]:
        """
        Get unique attack types from CIC-DDoS2019 dataset
        
        The dataset contains 11 types of DDoS attacks according to the paper:
        - Reflective DDoS: DNS, LDAP, MSSQL, TFTP
        - UDP, UDP-Lag, SYN
        - And other attack variants
        
        Args:
            df: DataFrame containing CIC-DDoS2019 data
            label_col: Label column name (auto-detected if None)
            
        Returns:
            List of unique attack type labels found in the dataset
        """
        if label_col is None:
            # Try to find label column
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
        """
        Get information about a dataset
        
        For CIC-DDoS2019, validates that dataset contains approximately 80 features
        as expected from CICFlowMeter extraction.
        
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'name': dataset_name,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Check for label column with improved detection
        label_candidates = ['Label', 'label', 'Attack', 'attack', 'Class', 'class']
        label_col = None
        for col in label_candidates:
            if col in df.columns:
                label_col = col
                info['label_column'] = col
                if df[col].dtype in ['object', 'category']:
                    info['unique_labels'] = df[col].value_counts().to_dict()
                else:
                    info['unique_labels'] = df[col].value_counts().to_dict()
                break
        
        # CIC-DDoS2019 specific validations
        if 'CIC-DDoS2019' in dataset_name or 'cic' in dataset_name.lower():
            feature_count = len([c for c in df.columns if c != label_col])
            info['feature_count'] = feature_count
            info['expected_features'] = 80  # CICFlowMeter standard
            
            # Warning if feature count differs significantly from 80
            if abs(feature_count - 80) > 10:
                import warnings
                warnings.warn(
                    f"CIC-DDoS2019 dataset has {feature_count} features, expected ~80 from CICFlowMeter. "
                    f"Difference: {abs(feature_count - 80)} features.",
                    UserWarning
                )
            
            # Get attack types
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
    
    # Test CIC-DDoS2019 loading (will fail if not downloaded)
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
