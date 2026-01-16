#!/usr/bin/env python3
"""
Dataset loader for CIC-DDoS2019 and TON_IoT datasets
Handles downloading and loading of datasets for the IRP research project
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


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
        
        print(f"Loading TON_IoT dataset from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"TON_IoT loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def load_cic_ddos2019(self, dataset_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load CIC-DDoS2019 dataset
        
        Args:
            dataset_path: Path to CIC-DDoS2019 directory or CSV file
            
        Returns:
            DataFrame containing CIC-DDoS2019 data
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
        
        print(f"Found {len(csv_files)} CSV files in CIC-DDoS2019 dataset")
        
        # Load and concatenate all CSV files
        dataframes = []
        for csv_file in csv_files:
            try:
                print(f"Loading {csv_file.name}...")
                df = pd.read_csv(csv_file, low_memory=False)
                dataframes.append(df)
            except Exception as e:
                print(f"Warning: Could not load {csv_file.name}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("No valid CSV files could be loaded from CIC-DDoS2019")
        
        # Concatenate all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"CIC-DDoS2019 loaded: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
        
        return combined_df
    
    def get_dataset_info(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """
        Get information about a dataset
        
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
        
        # Check for label column
        label_candidates = ['label', 'Label', 'Label', 'Class', 'class', 'Attack', 'attack']
        for col in label_candidates:
            if col in df.columns:
                info['label_column'] = col
                if df[col].dtype in ['object', 'category']:
                    info['unique_labels'] = df[col].value_counts().to_dict()
                else:
                    info['unique_labels'] = df[col].value_counts().to_dict()
                break
        
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
