#!/usr/bin/env python3
"""
Data harmonization module for CIC-DDoS2019 and TON_IoT datasets
Implements feature mapping and early fusion with statistical validation
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class DataHarmonizer:
    """Harmonizes heterogeneous datasets for joint analysis"""
    
    # CICFlowMeter feature patterns (to detect standard features)
    # These are common patterns found in CIC-DDoS2019 dataset (80 features)
    CICFLOWMETER_PATTERNS = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        'Fwd Packet Length', 'Bwd Packet Length', 'Flow Bytes/s',
        'Flow Packets/s', 'Flow IAT', 'Fwd IAT', 'Bwd IAT',
        'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
        'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
        'Packet Length', 'FIN Flag', 'SYN Flag', 'RST Flag', 'PSH Flag',
        'ACK Flag', 'URG Flag', 'CWE Flag', 'ECE Flag', 'Down/Up Ratio',
        'Average Packet', 'Avg Fwd Segment', 'Avg Bwd Segment',
        'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
        'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
        'Subflow Fwd', 'Subflow Bwd', 'Init_Win_bytes', 'Active', 'Idle'
    ]
    
    def __init__(self):
        """Initialize the harmonizer with feature mapping schemas"""
        self.feature_mapping_cic = {}
        self.feature_mapping_ton = {}
        self.harmonized_features = []
        self.statistical_tests = {}
        self.common_features_found = []
        
    def analyze_feature_similarity(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                                   col1: str, col2: str) -> Dict:
        """
        Analyze statistical similarity between two features
        
        Args:
            df1: First dataframe
            df2: Second dataframe
            col1: Column name in df1
            col2: Column name in df2
            
        Returns:
            Dictionary with statistical test results
        """
        # Get numeric values only
        val1 = pd.to_numeric(df1[col1], errors='coerce').dropna()
        val2 = pd.to_numeric(df2[col2], errors='coerce').dropna()
        
        if len(val1) == 0 or len(val2) == 0:
            return {'compatible': False, 'reason': 'Non-numeric or empty'}
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(val1, val2)
        
        # Additional statistics
        mean_diff = abs(val1.mean() - val2.mean())
        std_diff = abs(val1.std() - val2.std())
        
        # Consider compatible if p-value > 0.05 or distributions are similar
        compatible = ks_pvalue > 0.05 or (mean_diff < 0.1 * (val1.mean() + val2.mean()) / 2)
        
        return {
            'compatible': compatible,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'mean1': val1.mean(),
            'mean2': val2.mean()
        }
    
    def find_common_features(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                            df1_name: str = "Dataset1", df2_name: str = "Dataset2") -> Dict:
        """
        Find semantically similar features between datasets using intelligent analysis
        
        Args:
            df1: First dataframe (e.g., CIC-DDoS2019)
            df2: Second dataframe (e.g., TON_IoT)
            df1_name: Name of first dataset
            df2_name: Name of second dataset
            
        Returns:
            Dictionary mapping features
        """
        mapping = {}
        
        # Try to use FeatureAnalyzer if available for intelligent matching
        try:
            from feature_analyzer import FeatureAnalyzer
            analyzer = FeatureAnalyzer()
            common_features = analyzer.extract_common_features(df1, df2)
            
            # Convert to mapping format
            for feat in common_features:
                unified_name = feat['unified_name']
                mapping[unified_name] = {
                    'cic': feat['cic_name'],
                    'ton': feat['ton_name'],
                    'type': feat['type'],
                    'category': feat.get('category'),
                    'unit': feat.get('unit'),
                    'confidence': feat.get('confidence', 'medium')
                }
            
            logger.info(f"Found {len(mapping)} common features using FeatureAnalyzer")
            return mapping
            
        except ImportError:
            logger.warning("FeatureAnalyzer not available, using basic matching")
        except Exception as e:
            logger.warning(f"FeatureAnalyzer failed: {e}, using basic matching")
        
        # Fallback to basic matching
        # Common exact matches
        common_exact = set(df1.columns) & set(df2.columns)
        for col in common_exact:
            mapping[col] = {'cic': col, 'ton': col, 'type': 'exact_match'}
        
        # Enhanced semantic mappings based on actual analysis
        semantic_mappings = {
            # IP addresses
            'src_ip': ['Src IP', 'src_ip', 'Source IP', 'source_ip', 'srcip'],
            'dst_ip': ['Dst IP', 'dst_ip', 'Destination IP', 'destination_ip', 'dstip'],
            # Ports
            'src_port': ['Src Port', 'src_port', 'Source Port', 'source_port', 'srcport'],
            'dst_port': ['Dst Port', 'dst_port', 'Destination Port', 'destination_port', 'dstport'],
            # Protocol
            'proto': ['Protocol', 'proto', 'Protocol Name', 'protocol'],
            # Duration/Time
            'duration': ['Flow Duration', 'duration', 'Flow Duration (ms)', 'flow_duration', 'time'],
            # Packets - Forward
            'fwd_packets': ['Total Fwd Packets', 'fwd_packets', 'forward_packets', 'fwd_pkt'],
            'fwd_bytes': ['Total Length of Fwd Packets', 'fwd_bytes', 'forward_bytes', 'fwd_length'],
            # Packets - Backward
            'bwd_packets': ['Total Backward Packets', 'bwd_packets', 'backward_packets', 'bwd_pkt'],
            'bwd_bytes': ['Total Length of Bwd Packets', 'bwd_bytes', 'backward_bytes', 'bwd_length'],
            # Flow statistics
            'flow_bytes_per_sec': ['Flow Bytes/s', 'flow_bytes_per_sec', 'bytes_per_second', 'throughput'],
            'flow_packets_per_sec': ['Flow Packets/s', 'flow_packets_per_sec', 'packets_per_second'],
            # Flags
            'syn_flags': ['SYN Flag Count', 'syn_flag', 'syn_count', 'syn'],
            'ack_flags': ['ACK Flag Count', 'ack_flag', 'ack_count', 'ack'],
            'fin_flags': ['FIN Flag Count', 'fin_flag', 'fin_count', 'fin'],
        }
        
        for unified_name, variants in semantic_mappings.items():
            cic_match = None
            ton_match = None
            
            # Try exact match first
            for variant in variants:
                if variant in df1.columns and cic_match is None:
                    cic_match = variant
                if variant in df2.columns and ton_match is None:
                    ton_match = variant
            
            # Try case-insensitive and partial match
            if not cic_match:
                for col in df1.columns:
                    col_lower = col.lower()
                    for variant in variants:
                        if variant.lower() in col_lower or col_lower in variant.lower():
                            cic_match = col
                            break
                    if cic_match:
                        break
            
            if not ton_match:
                for col in df2.columns:
                    col_lower = col.lower()
                    for variant in variants:
                        if variant.lower() in col_lower or col_lower in variant.lower():
                            ton_match = col
                            break
                    if ton_match:
                        break
            
            if cic_match and ton_match:
                mapping[unified_name] = {
                    'cic': cic_match,
                    'ton': ton_match,
                    'type': 'semantic_match'
                }
        
        return mapping
    
    def harmonize_features(self, df_cic: pd.DataFrame, df_ton: pd.DataFrame,
                          label_col_cic: Optional[str] = None,
                          label_col_ton: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Harmonize features from both datasets to a common schema
        
        Args:
            df_cic: CIC-DDoS2019 dataframe
            df_ton: TON_IoT dataframe
            label_col_cic: Label column name in CIC-DDoS2019
            label_col_ton: Label column name in TON_IoT (default: 'label')
            
        Returns:
            Tuple of harmonized dataframes (df_cic_harmonized, df_ton_harmonized)
        """
        # Auto-detect label columns
        if label_col_cic is None:
            for col in ['Label', 'label', 'Attack', 'Class']:
                if col in df_cic.columns:
                    label_col_cic = col
                    break
        
        if label_col_ton is None:
            label_col_ton = 'label' if 'label' in df_ton.columns else None
        
        # Find common features
        feature_mapping = self.find_common_features(df_cic, df_ton, "CIC-DDoS2019", "TON_IoT")
        self.common_features_found = list(feature_mapping.keys())
        
        # Use CICFlowMeter standard features if available in CIC dataset
        # Detect CICFlowMeter features by pattern matching (more flexible than exact match)
        cic_standard_features = []
        for pattern in self.CICFLOWMETER_PATTERNS:
            matching = [col for col in df_cic.columns if pattern.lower() in col.lower()]
            cic_standard_features.extend(matching)
        
        # Remove duplicates while preserving order
        cic_standard_features = list(dict.fromkeys(cic_standard_features))
        
        if len(cic_standard_features) > 0:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"   Found {len(cic_standard_features)} CICFlowMeter-like features in CIC-DDoS2019")
            # Try to map TON_IoT features to CICFlowMeter features
            for cic_feat in cic_standard_features:
                if cic_feat not in feature_mapping:
                    # Try to find semantic match in TON_IoT
                    cic_lower = cic_feat.lower().replace(' ', '_').replace('/', '_')
                    for ton_col in df_ton.columns:
                        ton_lower = ton_col.lower().replace(' ', '_').replace('/', '_')
                        # More flexible matching
                        if (cic_lower in ton_lower or ton_lower in cic_lower or 
                            any(term in ton_lower for term in cic_lower.split('_') if len(term) > 3)):
                            feature_mapping[cic_feat] = {
                                'cic': cic_feat,
                                'ton': ton_col,
                                'type': 'cicflowmeter_standard'
                            }
                            break
        
        # Extract numeric features that can be harmonized
        harmonized_cols = []
        cic_data = {}
        ton_data = {}
        
        for unified_name, mapping_info in feature_mapping.items():
            cic_col = mapping_info['cic']
            ton_col = mapping_info['ton']
            
            # Convert to numeric
            cic_values = pd.to_numeric(df_cic[cic_col], errors='coerce')
            ton_values = pd.to_numeric(df_ton[ton_col], errors='coerce')
            
            if cic_values.notna().sum() > 0.8 * len(df_cic) and \
               ton_values.notna().sum() > 0.8 * len(df_ton):
                harmonized_cols.append(unified_name)
                cic_data[unified_name] = cic_values.fillna(cic_values.median())
                ton_data[unified_name] = ton_values.fillna(ton_values.median())
        
        # Create harmonized dataframes
        df_cic_harmonized = pd.DataFrame(cic_data)
        df_ton_harmonized = pd.DataFrame(ton_data)
        
        # Add labels
        if label_col_cic:
            # Convert label to binary if needed
            cic_labels = df_cic[label_col_cic].copy()
            if cic_labels.dtype == 'object':
                # Map attack/normal to 1/0
                unique_labels = cic_labels.unique()
                cic_labels = (cic_labels != unique_labels[0]).astype(int)
            df_cic_harmonized['label'] = pd.to_numeric(cic_labels, errors='coerce').fillna(0)
        
        if label_col_ton:
            ton_labels = df_ton[label_col_ton].copy()
            if ton_labels.dtype == 'object':
                unique_labels = ton_labels.unique()
                ton_labels = (ton_labels != unique_labels[0]).astype(int)
            df_ton_harmonized['label'] = pd.to_numeric(ton_labels, errors='coerce').fillna(0)
        
        self.harmonized_features = harmonized_cols
        self.feature_mapping_cic = {v['cic']: k for k, v in feature_mapping.items()}
        self.feature_mapping_ton = {v['ton']: k for k, v in feature_mapping.items()}
        
        return df_cic_harmonized, df_ton_harmonized
    
    def early_fusion(self, df_cic_harmonized: pd.DataFrame, 
                    df_ton_harmonized: pd.DataFrame,
                    validate: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform early fusion of harmonized datasets with statistical validation
        
        Args:
            df_cic_harmonized: Harmonized CIC-DDoS2019 dataframe
            df_ton_harmonized: Harmonized TON_IoT dataframe
            validate: Whether to perform Kolmogorov-Smirnov validation
            
        Returns:
            Tuple of (fused_dataframe, validation_results)
        """
        # Ensure same columns
        common_cols = set(df_cic_harmonized.columns) & set(df_ton_harmonized.columns)
        common_cols = sorted(list(common_cols))
        
        df_cic_aligned = df_cic_harmonized[common_cols]
        df_ton_aligned = df_ton_harmonized[common_cols]
        
        # Add dataset source indicator
        df_cic_aligned['dataset_source'] = 'CIC-DDoS2019'
        df_ton_aligned['dataset_source'] = 'TON_IoT'
        
        # Concatenate
        df_fused = pd.concat([df_cic_aligned, df_ton_aligned], ignore_index=True)
        
        validation_results = {}
        
        if validate:
            # Perform KS tests on common numeric features (excluding label and source)
            numeric_features = [col for col in common_cols 
                              if col not in ['label', 'dataset_source'] and
                              pd.api.types.is_numeric_dtype(df_fused[col])]
            
            for feature in numeric_features[:10]:  # Limit to first 10 for performance
                cic_values = df_cic_aligned[feature].dropna()
                ton_values = df_ton_aligned[feature].dropna()
                
                if len(cic_values) > 0 and len(ton_values) > 0:
                    ks_stat, ks_pvalue = stats.ks_2samp(cic_values, ton_values)
                    validation_results[feature] = {
                        'ks_statistic': ks_stat,
                        'ks_pvalue': ks_pvalue,
                        'compatible': ks_pvalue > 0.05,
                        'cic_mean': cic_values.mean(),
                        'ton_mean': ton_values.mean()
                    }
            
            self.statistical_tests = validation_results
        
        return df_fused, validation_results
    
    def get_harmonization_report(self) -> str:
        """Generate a report on the harmonization process"""
        report = f"Harmonization Report\n"
        report += f"=" * 50 + "\n"
        report += f"Number of harmonized features: {len(self.harmonized_features)}\n"
        report += f"\nHarmonized features: {', '.join(self.harmonized_features)}\n"
        
        if self.statistical_tests:
            report += f"\nStatistical Validation (Kolmogorov-Smirnov):\n"
            compatible = sum(1 for v in self.statistical_tests.values() if v.get('compatible', False))
            total = len(self.statistical_tests)
            report += f"  Compatible features: {compatible}/{total}\n"
            
            for feature, results in list(self.statistical_tests.items())[:5]:
                report += f"  {feature}: p-value={results['ks_pvalue']:.4f}, "
                report += f"compatible={results['compatible']}\n"
        
        return report


def main():
    """Test the harmonization module"""
    from dataset_loader import DatasetLoader
    
    loader = DatasetLoader()
    
    # Load datasets
    try:
        df_ton = loader.load_ton_iot()
        print(f"TON_IoT loaded: {df_ton.shape}\n")
    except Exception as e:
        print(f"Could not load TON_IoT: {e}")
        return
    
    try:
        df_cic = loader.load_cic_ddos2019()
        print(f"CIC-DDoS2019 loaded: {df_cic.shape}\n")
    except Exception as e:
        print(f"CIC-DDoS2019 not available: {e}")
        print("Harmonization test skipped.")
        return
    
    # Harmonize
    harmonizer = DataHarmonizer()
    df_cic_harm, df_ton_harm = harmonizer.harmonize_features(df_cic, df_ton)
    
    print(f"CIC-DDoS2019 harmonized: {df_cic_harm.shape}")
    print(f"TON_IoT harmonized: {df_ton_harm.shape}\n")
    
    # Early fusion
    df_fused, validation = harmonizer.early_fusion(df_cic_harm, df_ton_harm)
    print(f"Fused dataset: {df_fused.shape}\n")
    
    # Report
    print(harmonizer.get_harmonization_report())


if __name__ == "__main__":
    main()
