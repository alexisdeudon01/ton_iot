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
import logging

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DataHarmonizer:
    """Harmonizes heterogeneous datasets for joint analysis"""
    
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
            
            # Log common features details
            if mapping:
                logger.info(f"[FEATURES COMMUNES] Liste des {len(mapping)} features trouvées:")
                for i, (unified_name, info) in enumerate(mapping.items(), 1):
                    cic_name = info.get('cic', 'N/A')
                    ton_name = info.get('ton', 'N/A')
                    feat_type = info.get('type', 'unknown')
                    confidence = info.get('confidence', 'unknown')
                    logger.info(f"  {i:2d}. {unified_name:30s} | CIC: {cic_name:35s} | TON: {ton_name:35s} | Type: {feat_type:15s} | Conf: {confidence}")
            
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
        
        logger.info(f"[FEATURES COMMUNES] {len(common_exact)} features exactes trouvées (fallback)")
        if common_exact:
            logger.info(f"  Features exactes: {', '.join(sorted(list(common_exact))[:10])}{'...' if len(common_exact) > 10 else ''}")
        
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
        
        if mapping:
            logger.info(f"[FEATURES COMMUNES] Total {len(mapping)} features communes (exactes + sémantiques)")
            # Log summary
            exact_count = sum(1 for info in mapping.values() if info.get('type') == 'exact_match')
            semantic_count = sum(1 for info in mapping.values() if info.get('type') == 'semantic_match')
            logger.info(f"  - Exactes: {exact_count}")
            logger.info(f"  - Sémantiques: {semantic_count}")
        
        return mapping
    
    def harmonize_features(self, df_cic: pd.DataFrame, df_ton: pd.DataFrame,
                          label_col_cic: Optional[str] = None,
                          label_col_ton: Optional[str] = None,
                          precomputed_feature_mapping: Optional[Dict] = None,
                          filter_ton_by_type: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Harmonize features from both datasets to a common schema
        
        Args:
            df_cic: CIC-DDoS2019 dataframe
            df_ton: TON_IoT dataframe
            label_col_cic: Label column name in CIC-DDoS2019 (if None, uses last column)
            label_col_ton: Label column name in TON_IoT (if None, uses last column)
            precomputed_feature_mapping: Pre-computed feature mapping from pre-analysis (optional)
            filter_ton_by_type: If True, filter TON_IoT to keep only rows with type='normal' or 'ddos'
            
        Returns:
            Tuple of harmonized dataframes (df_cic_harmonized, df_ton_harmonized)
        """
        # Auto-detect label columns (use last column as per requirements)
        if label_col_cic is None:
            label_col_cic = df_cic.columns[-1]
            logger.info(f"[LABEL] CIC-DDoS2019: Using last column as label: {label_col_cic}")
        
        if label_col_ton is None:
            label_col_ton = df_ton.columns[-1]
            logger.info(f"[LABEL] TON_IoT: Using last column as label: {label_col_ton}")
        
        # Filter TON_IoT: keep only rows with type='normal' or 'ddos' (if filter_ton_by_type is True)
        if filter_ton_by_type and 'type' in df_ton.columns:
            original_size = len(df_ton)
            valid_mask = df_ton['type'].isin(['normal', 'ddos'])
            df_ton = df_ton.loc[valid_mask].copy()
            logger.info(f"[FILTER] TON_IoT: Kept {len(df_ton)}/{original_size} rows (type='normal' or 'ddos')")
            if len(df_ton) == 0:
                raise ValueError("No valid rows in TON_IoT after filtering (type='normal' or 'ddos')")
        
        # Use precomputed feature mapping if available, otherwise find common features
        if precomputed_feature_mapping is not None:
            logger.info("   Using pre-computed feature mapping from pre-analysis")
            # Convert precomputed format (list of dicts) to mapping format (dict)
            feature_mapping = {}
            for feat in precomputed_feature_mapping:
                unified_name = feat.get('unified_name', feat.get('cic_name'))
                if unified_name:
                    feature_mapping[unified_name] = {
                        'cic': feat.get('cic_name', unified_name),
                        'ton': feat.get('ton_name', unified_name),
                        'type': feat.get('type', 'unknown'),
                        'category': feat.get('category'),
                        'unit': feat.get('unit'),
                        'confidence': feat.get('confidence', 'medium')
                    }
            logger.info(f"   Loaded {len(feature_mapping)} features from pre-analysis")
        else:
            # Find common features (fallback)
            logger.info("   Computing feature mapping (no precomputed mapping provided)")
            feature_mapping = self.find_common_features(df_cic, df_ton, "CIC-DDoS2019", "TON_IoT")
        
        self.common_features_found = list(feature_mapping.keys())
        
        # Log IRP requirements
        try:
            from irp_features_requirements import IRPFeaturesRequirements
            irp_req = IRPFeaturesRequirements()
            logger.info("[IRP FEATURES] Exigences selon IRP_EXACT_FORMULAS.md:")
            logger.info("  • Dimension 1: Toutes features numériques + label (pour entraîner modèles)")
            logger.info("  • Dimension 2: Aucune feature dataset (métriques runtime uniquement)")
            logger.info("  • Dimension 3: Toutes features numériques + feature names (pour SHAP/LIME)")
            logger.info("  → Conclusion: Toutes les features numériques communes harmonisées + label")
        except Exception as e:
            logger.debug(f"IRP features requirements not available: {e}")
        
        # Extract numeric features that can be harmonized
        # According to IRP_EXACT_FORMULAS.md: We need ALL numeric features harmonized
        harmonized_cols = []
        cic_data = {}
        ton_data = {}
        
        # Filter feature_mapping to only include features that actually exist in both DataFrames
        valid_feature_mapping = {}
        for unified_name, mapping_info in feature_mapping.items():
            cic_col = mapping_info.get('cic')
            ton_col = mapping_info.get('ton')
            
            # Verify columns exist in actual DataFrames
            if cic_col not in df_cic.columns:
                logger.debug(f"[HARMONIZATION] Skipping '{unified_name}': CIC column '{cic_col}' not found in DataFrame")
                continue
            if ton_col not in df_ton.columns:
                logger.debug(f"[HARMONIZATION] Skipping '{unified_name}': TON column '{ton_col}' not found in DataFrame")
                continue
            
            valid_feature_mapping[unified_name] = mapping_info
        
        logger.info(f"[HARMONIZATION] Using {len(valid_feature_mapping)} valid features (from {len(feature_mapping)} mapped)")
        
        for unified_name, mapping_info in valid_feature_mapping.items():
            cic_col = mapping_info['cic']
            ton_col = mapping_info['ton']
            
            # Convert to numeric (IRP requires numeric features for model training)
            try:
                cic_values = pd.to_numeric(df_cic[cic_col], errors='coerce')
                ton_values = pd.to_numeric(df_ton[ton_col], errors='coerce')
            except KeyError as e:
                logger.warning(f"[HARMONIZATION] Column not found: {e}, skipping feature '{unified_name}'")
                continue
            
            # Keep features with at least 80% valid numeric values
            # These are necessary for IRP calculations (Dimension 1 & 3 require trained models)
            if cic_values.notna().sum() > 0.8 * len(df_cic) and \
               ton_values.notna().sum() > 0.8 * len(df_ton):
                harmonized_cols.append(unified_name)
                cic_data[unified_name] = cic_values.fillna(cic_values.median())
                ton_data[unified_name] = ton_values.fillna(ton_values.median())
        
        logger.info(f"[IRP FEATURES] {len(harmonized_cols)} features numériques harmonisées (nécessaires pour calculs IRP)")
        
        # Create harmonized dataframes
        df_cic_harmonized = pd.DataFrame(cic_data)
        df_ton_harmonized = pd.DataFrame(ton_data)
        
        # Add labels with binary classification
        # CIC-DDoS2019: After concatenating ALL files, use last column (Label)
        # Everything that is NOT 'Benign' in Label column = Attack (1)
        if label_col_cic and label_col_cic in df_cic.columns:
            # df_cic is already the concatenated result of ALL CIC-DDoS2019 files
            cic_labels = df_cic[label_col_cic].copy()
            logger.info(f"[LABEL] CIC-DDoS2019: Processing labels from concatenated dataset ({len(cic_labels)} rows)")
            
            if cic_labels.dtype == 'object':
                # Binary classification: Benign = 0, all attacks (NOT 'Benign') = 1
                # Check all unique values to confirm
                unique_labels = cic_labels.unique()
                logger.debug(f"  Unique label values in last column: {sorted(unique_labels)}")
                
                # Everything that is NOT 'Benign' (case-insensitive) = Attack (1)
                cic_labels_binary = (cic_labels.str.upper() != 'BENIGN').astype(int)
                
                logger.info(f"[LABEL] CIC-DDoS2019 binary classification: Benign=0, Attacks (non-Benign)=1")
                logger.info(f"  Total rows: {len(cic_labels)}")
                logger.info(f"  Original label distribution: {cic_labels.value_counts().to_dict()}")
                logger.info(f"  Binary distribution: {cic_labels_binary.value_counts().to_dict()}")
                logger.info(f"  → Benign (0): {len(cic_labels_binary[cic_labels_binary == 0])}, Attacks (1): {len(cic_labels_binary[cic_labels_binary == 1])}")
            else:
                # Already numeric, assume 0=Benign, 1=Attack
                cic_labels_binary = pd.to_numeric(cic_labels, errors='coerce').fillna(0).astype(int)
                logger.info(f"[LABEL] CIC-DDoS2019: Labels already numeric, using as-is (0=Benign, 1=Attack)")
            
            df_cic_harmonized['label'] = cic_labels_binary
        
        # TON_IoT: Binary classification: normal=0, ddos=1
        # Note: Filtering by type was done earlier if filter_ton_by_type=True
        if label_col_ton and label_col_ton in df_ton.columns:
            ton_labels = df_ton[label_col_ton].copy()
            
            # Use 'type' column if available for binary classification
            if 'type' in df_ton.columns:
                # Binary classification based on type column: normal = 0, ddos = 1
                ton_labels_binary = (df_ton['type'].str.lower() == 'ddos').astype(int)
                logger.info(f"[LABEL] TON_IoT binary classification (using 'type' column): normal=0, ddos=1")
                logger.info(f"  Type distribution: {df_ton['type'].value_counts().to_dict()}")
            else:
                # Fallback: use label column directly
                if ton_labels.dtype == 'object':
                    ton_labels_binary = (ton_labels.str.lower() == 'ddos').astype(int)
                else:
                    ton_labels_binary = pd.to_numeric(ton_labels, errors='coerce').fillna(0).astype(int)
                logger.warning(f"[LABEL] TON_IoT 'type' column not found, using label column directly")
            
            logger.info(f"  Binary distribution: {ton_labels_binary.value_counts().to_dict()}")
            df_ton_harmonized['label'] = ton_labels_binary
        
        self.harmonized_features = harmonized_cols
        self.feature_mapping_cic = {v['cic']: k for k, v in feature_mapping.items()}
        self.feature_mapping_ton = {v['ton']: k for k, v in feature_mapping.items()}
        
        # Verify normalization of harmonized features
        self._verify_harmonized_normalization(df_cic_harmonized, df_ton_harmonized)
        
        return df_cic_harmonized, df_ton_harmonized
    
    def _verify_harmonized_normalization(self, df_cic: pd.DataFrame, df_ton: pd.DataFrame) -> Dict:
        """
        Verify that harmonized features are properly normalized (after preprocessing)
        Checks min/max values and statistical distribution
        
        Args:
            df_cic: Harmonized CIC-DDoS2019 DataFrame
            df_ton: Harmonized TON_IoT DataFrame
            
        Returns:
            Dictionary with verification results
        """
        verification_results = {
            'cic_stats': {},
            'ton_stats': {},
            'normalized': True,
            'issues': []
        }
        
        # Check each harmonized feature
        common_features = set(df_cic.columns) & set(df_ton.columns)
        common_features = {f for f in common_features if f != 'label'}
        
        if not common_features:
            logger.warning("[VERIFICATION] No common features to verify normalization")
            return verification_results
        
        logger.info(f"[VERIFICATION] Vérification normalisation pour {len(common_features)} features communes...")
        
        for feat in sorted(common_features):
            cic_values = df_cic[feat].dropna()
            ton_values = df_ton[feat].dropna()
            
            if len(cic_values) == 0 or len(ton_values) == 0:
                continue
            
            # Calculate statistics
            cic_stats = {
                'min': float(cic_values.min()),
                'max': float(cic_values.max()),
                'mean': float(cic_values.mean()),
                'std': float(cic_values.std()),
                'range': float(cic_values.max() - cic_values.min())
            }
            
            ton_stats = {
                'min': float(ton_values.min()),
                'max': float(ton_values.max()),
                'mean': float(ton_values.mean()),
                'std': float(ton_values.std()),
                'range': float(ton_values.max() - ton_values.min())
            }
            
            verification_results['cic_stats'][feat] = cic_stats
            verification_results['ton_stats'][feat] = ton_stats
            
            # Check if values are in reasonable range (after RobustScaler, values are typically in [-3, 3] for IQR-based)
            # But also check if they could be min-max scaled [0, 1]
            cic_in_01_range = (cic_stats['min'] >= -0.1 and cic_stats['max'] <= 1.1)
            ton_in_01_range = (ton_stats['min'] >= -0.1 and ton_stats['max'] <= 1.1)
            
            cic_in_robust_range = (cic_stats['min'] >= -5.0 and cic_stats['max'] <= 5.0)
            ton_in_robust_range = (ton_stats['min'] >= -5.0 and ton_stats['max'] <= 5.0)
            
            # Log if feature is already normalized (either min-max or robust)
            if cic_in_01_range and ton_in_01_range:
                logger.debug(f"  ✓ {feat}: Normalisé min-max [0, 1] | CIC: [{cic_stats['min']:.3f}, {cic_stats['max']:.3f}] | TON: [{ton_stats['min']:.3f}, {ton_stats['max']:.3f}]")
            elif cic_in_robust_range and ton_in_robust_range:
                logger.debug(f"  ✓ {feat}: Normalisé robuste [-5, 5] | CIC: [{cic_stats['min']:.3f}, {cic_stats['max']:.3f}] | TON: [{ton_stats['min']:.3f}, {ton_stats['max']:.3f}]")
            else:
                # Feature not normalized yet (will be normalized in preprocessing pipeline)
                logger.debug(f"  ⚠ {feat}: Non normalisé (sera normalisé dans preprocessing) | CIC: [{cic_stats['min']:.2f}, {cic_stats['max']:.2f}] | TON: [{ton_stats['min']:.2f}, {ton_stats['max']:.2f}]")
        
        logger.info(f"[VERIFICATION] Vérification terminée: {len(common_features)} features analysées")
        logger.info(f"[VERIFICATION] Note: Les features harmonisées seront normalisées avec RobustScaler dans preprocessing_pipeline")
        
        return verification_results
    
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
