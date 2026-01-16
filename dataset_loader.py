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

warnings.filterwarnings('ignore')

# Setup logger
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loader for CIC-DDoS2019 and TON_IoT datasets"""
    
    def __init__(self, data_dir: str = 'data/raw'):
        """
        Initialize the dataset loader
        
        Args:
            data_dir: Root directory for raw datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_ton_iot(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load TON_IoT dataset
        
        Args:
            file_path: Path to TON_IoT CSV file. If None, looks for train_test_network.csv in root
            
        Returns:
            DataFrame containing TON_IoT data
        """
        if file_path is None:
            # Try to find train_test_network.csv in current directory or data/raw
            possible_paths = [
                Path('train_test_network.csv'),
                self.data_dir / 'TON_IoT' / 'train_test_network.csv',
                Path('Processed_datasets/Processed_Windows_dataset/windows10_dataset.csv')
            ]
            
            for path in possible_paths:
                if path.exists():
                    file_path = str(path)
                    break
            
            if file_path is None:
                raise FileNotFoundError(
                    "TON_IoT dataset not found. Please provide file_path or place "
                    "train_test_network.csv in the project root or data/raw/TON_IoT/"
                )
        
        logger.info(f"Loading TON_IoT dataset from: {file_path}")
        try:
            df = pd.read_csv(file_path)
            logger.info(f"TON_IoT loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading TON_IoT dataset: {e}", exc_info=True)
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
            dataset_path = self.data_dir / 'CIC-DDoS2019'
        
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"CIC-DDoS2019 dataset not found at {dataset_path}. "
                "Please download the dataset from: "
                "https://www.unb.ca/cic/datasets/ddos-2019.html "
                "and place it in data/raw/CIC-DDoS2019/"
            )
        
        # CIC-DDoS2019 typically contains multiple CSV files (one per attack type)
        csv_files = list(dataset_path.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {dataset_path}")
        
        logger.info(f"Found {len(csv_files)} CSV files in CIC-DDoS2019 dataset")
        
        # Load and concatenate all CSV files with progress bar
        dataframes = []
        for csv_file in tqdm(csv_files, desc="Loading CIC-DDoS2019 CSV files"):
            try:
                logger.debug(f"Loading {csv_file.name}...")
                df = pd.read_csv(csv_file, low_memory=False)
                dataframes.append(df)
                logger.debug(f"Loaded {csv_file.name}: {df.shape}")
            except Exception as e:
                logger.warning(f"Could not load {csv_file.name}: {e}")
                continue
        
        if not dataframes:
            error_msg = "No valid CSV files could be loaded from CIC-DDoS2019"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Concatenate all dataframes
        logger.info("Concatenating all CSV files...")
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"CIC-DDoS2019 loaded: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
        
        return combined_df
    
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
