#!/usr/bin/env python3
"""
Feature Analyzer for TON_IoT and CIC-DDoS2019 datasets
Analyzes actual CSV columns, units, and data types to propose common features
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import logging
import re

logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """Analyze features from datasets to find common mappings"""
    
    def __init__(self):
        """Initialize feature analyzer"""
        self.ton_features = {}
        self.cic_features = {}
        self.common_features_proposed = []
    
    def analyze_dataset_features(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """
        Analyze features from a dataset
        
        Args:
            df: DataFrame to analyze
            dataset_name: Name of dataset ('TON_IoT' or 'CIC-DDoS2019')
            
        Returns:
            Dictionary with feature analysis
        """
        analysis = {
            'dataset_name': dataset_name,
            'total_features': len(df.columns),
            'numeric_features': [],
            'categorical_features': [],
            'feature_info': {}
        }
        
        for col in df.columns:
            col_lower = col.lower().strip()
            dtype = df[col].dtype
            
            # Sample values to understand units
            sample_values = df[col].dropna().head(10)
            
            # Detect feature type and units
            feature_info = {
                'name': col,
                'dtype': str(dtype),
                'is_numeric': pd.api.types.is_numeric_dtype(df[col]),
                'has_units': False,
                'unit': None,
                'category': None,
                'sample_values': sample_values.tolist()[:5] if len(sample_values) > 0 else []
            }
            
            # Detect units from column name or values
            unit_patterns = {
                'bytes': ['byte', 'bytes', 'b'],
                'seconds': ['sec', 'second', 'seconds', 's', 'duration', 'time'],
                'milliseconds': ['ms', 'millisecond', 'milliseconds'],
                'microseconds': ['us', 'microsecond', 'microseconds'],
                'packets': ['packet', 'packets', 'pkt'],
                'count': ['count', 'cnt', 'num', 'number'],
                'ratio': ['ratio', 'rate', 'percent', '%'],
                'flags': ['flag', 'flags'],
                'ip': ['ip', 'address'],
                'port': ['port'],
                'protocol': ['protocol', 'proto']
            }
            
            for unit, patterns in unit_patterns.items():
                if any(pattern in col_lower for pattern in patterns):
                    feature_info['has_units'] = True
                    feature_info['unit'] = unit
                    break
            
            # Categorize feature
            feature_info['category'] = self._categorize_feature(col, col_lower, dtype, sample_values)
            
            analysis['feature_info'][col] = feature_info
            
            if pd.api.types.is_numeric_dtype(df[col]):
                analysis['numeric_features'].append(col)
            else:
                analysis['categorical_features'].append(col)
        
        return analysis
    
    def _categorize_feature(self, col_name: str, col_lower: str, dtype, sample_values: pd.Series) -> str:
        """Categorize a feature based on name and values"""
        
        # IP addresses
        if any(term in col_lower for term in ['ip', 'address', 'src_ip', 'dst_ip', 'source ip', 'destination ip']):
            return 'ip_address'
        
        # Ports
        if any(term in col_lower for term in ['port', 'src_port', 'dst_port', 'source port', 'destination port']):
            return 'port'
        
        # Protocol
        if any(term in col_lower for term in ['protocol', 'proto']):
            return 'protocol'
        
        # Duration/Time
        if any(term in col_lower for term in ['duration', 'time', 'iat', 'active', 'idle']):
            return 'temporal'
        
        # Packets
        if any(term in col_lower for term in ['packet', 'pkt', 'fwd packet', 'bwd packet']):
            return 'packet'
        
        # Bytes
        if any(term in col_lower for term in ['byte', 'bytes', 'length', 'size']):
            return 'byte'
        
        # Flags
        if any(term in col_lower for term in ['flag', 'fin', 'syn', 'rst', 'psh', 'ack', 'urg']):
            return 'flag'
        
        # Flow statistics
        if any(term in col_lower for term in ['flow', 'total', 'mean', 'std', 'max', 'min', 'avg']):
            return 'flow_statistic'
        
        # Rate/Throughput
        if any(term in col_lower for term in ['rate', 'per second', '/s', 'bytes/s', 'packets/s']):
            return 'rate'
        
        # Label
        if any(term in col_lower for term in ['label', 'attack', 'class', 'type']):
            return 'label'
        
        return 'other'
    
    def extract_common_features(self, df_ton: pd.DataFrame, df_cic: pd.DataFrame) -> List[Dict]:
        """
        Extract common features by analyzing both datasets
        
        Args:
            df_ton: TON_IoT DataFrame
            df_cic: CIC-DDoS2019 DataFrame
            
        Returns:
            List of common feature mappings
        """
        logger.info("Analyzing TON_IoT features...")
        ton_analysis = self.analyze_dataset_features(df_ton, 'TON_IoT')
        self.ton_features = ton_analysis['feature_info']
        
        logger.info("Analyzing CIC-DDoS2019 features...")
        cic_analysis = self.analyze_dataset_features(df_cic, 'CIC-DDoS2019')
        self.cic_features = cic_analysis['feature_info']
        
        logger.info("Extracting common features...")
        common_features = []
        
        # 1. Exact matches
        exact_matches = set(df_ton.columns) & set(df_cic.columns)
        for col in exact_matches:
            common_features.append({
                'unified_name': col,
                'ton_name': col,
                'cic_name': col,
                'type': 'exact_match',
                'category': ton_analysis['feature_info'][col]['category'],
                'unit': ton_analysis['feature_info'][col].get('unit'),
                'confidence': 'high'
            })
        
        # 2. Semantic matches by category
        ton_by_category = {}
        cic_by_category = {}
        
        for col, info in self.ton_features.items():
            cat = info['category']
            if cat not in ton_by_category:
                ton_by_category[cat] = []
            ton_by_category[cat].append((col, info))
        
        for col, info in self.cic_features.items():
            cat = info['category']
            if cat not in cic_by_category:
                cic_by_category[cat] = []
            cic_by_category[cat].append((col, info))
        
        # Match by category and similarity
        for category in set(ton_by_category.keys()) & set(cic_by_category.keys()):
            if category in ['label', 'ip_address', 'port', 'protocol']:
                continue  # Skip these, handled separately
            
            ton_cols = ton_by_category[category]
            cic_cols = cic_by_category[category]
            
            for ton_col, ton_info in ton_cols:
                if ton_col in exact_matches:
                    continue
                
                best_match = None
                best_score = 0
                
                for cic_col, cic_info in cic_cols:
                    if cic_col in exact_matches:
                        continue
                    
                    # Calculate similarity score
                    score = self._calculate_similarity(ton_col, cic_col, ton_info, cic_info)
                    if score > best_score and score > 0.5:  # Threshold
                        best_score = score
                        best_match = (cic_col, cic_info)
                
                if best_match:
                    cic_col, cic_info = best_match
                    unified_name = self._create_unified_name(ton_col, cic_col, category)
                    
                    common_features.append({
                        'unified_name': unified_name,
                        'ton_name': ton_col,
                        'cic_name': cic_col,
                        'type': 'semantic_match',
                        'category': category,
                        'unit': ton_info.get('unit') or cic_info.get('unit'),
                        'similarity_score': best_score,
                        'confidence': 'medium' if best_score > 0.7 else 'low'
                    })
        
        # 3. Specific mappings for known features
        specific_mappings = self._get_specific_mappings(df_ton, df_cic)
        for mapping in specific_mappings:
            # Check if not already added
            if not any(f['ton_name'] == mapping['ton_name'] and f['cic_name'] == mapping['cic_name'] 
                      for f in common_features):
                common_features.append(mapping)
        
        self.common_features_proposed = common_features
        return common_features
    
    def _calculate_similarity(self, name1: str, name2: str, info1: Dict, info2: Dict) -> float:
        """Calculate similarity score between two feature names"""
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        score = 0.0
        
        # Exact match (already handled, but check substring)
        if name1_lower == name2_lower:
            return 1.0
        
        # Word overlap
        words1 = set(re.findall(r'\w+', name1_lower))
        words2 = set(re.findall(r'\w+', name2_lower))
        if words1 and words2:
            overlap = len(words1 & words2) / len(words1 | words2)
            score += overlap * 0.4
        
        # Category match
        if info1.get('category') == info2.get('category'):
            score += 0.3
        
        # Unit match
        if info1.get('unit') and info2.get('unit'):
            if info1['unit'] == info2['unit']:
                score += 0.3
        
        return min(score, 1.0)
    
    def _create_unified_name(self, ton_name: str, cic_name: str, category: str) -> str:
        """Create a unified feature name"""
        # Try to use the shorter, more standard name
        if len(ton_name) <= len(cic_name):
            base = ton_name.lower().replace(' ', '_')
        else:
            base = cic_name.lower().replace(' ', '_')
        
        # Clean up
        base = re.sub(r'[^a-z0-9_]', '', base)
        return base
    
    def _get_specific_mappings(self, df_ton: pd.DataFrame, df_cic: pd.DataFrame) -> List[Dict]:
        """Get specific known mappings based on common network flow features"""
        mappings = []
        
        # Known mappings based on CICFlowMeter and TON_IoT structure
        known_mappings = [
            # Duration/Time
            {
                'unified_name': 'flow_duration',
                'ton_variants': ['duration', 'flow_duration', 'time', 'flow_time'],
                'cic_variants': ['Flow Duration', 'flow duration', 'Duration'],
                'category': 'temporal',
                'unit': 'microseconds'
            },
            # Packets - Forward
            {
                'unified_name': 'fwd_packets',
                'ton_variants': ['fwd_packets', 'forward_packets', 'src_packets', 'packets_sent', 'total_packets'],
                'cic_variants': ['Total Fwd Packets', 'total fwd packets', 'Fwd Packets'],
                'category': 'packet',
                'unit': 'count'
            },
            # Packets - Backward
            {
                'unified_name': 'bwd_packets',
                'ton_variants': ['bwd_packets', 'backward_packets', 'dst_packets', 'packets_received'],
                'cic_variants': ['Total Backward Packets', 'total backward packets', 'Bwd Packets'],
                'category': 'packet',
                'unit': 'count'
            },
            # Bytes - Forward
            {
                'unified_name': 'fwd_bytes',
                'ton_variants': ['fwd_bytes', 'forward_bytes', 'src_bytes', 'bytes_sent', 'fwd_length', 'total_bytes'],
                'cic_variants': ['Total Length of Fwd Packets', 'total length of fwd packets', 'Fwd Bytes'],
                'category': 'byte',
                'unit': 'bytes'
            },
            # Bytes - Backward
            {
                'unified_name': 'bwd_bytes',
                'ton_variants': ['bwd_bytes', 'backward_bytes', 'dst_bytes', 'bytes_received', 'bwd_length'],
                'cic_variants': ['Total Length of Bwd Packets', 'total length of bwd packets', 'Bwd Bytes'],
                'category': 'byte',
                'unit': 'bytes'
            },
            # Flow Rate - Bytes per second
            {
                'unified_name': 'flow_bytes_per_sec',
                'ton_variants': ['flow_bytes_per_sec', 'bytes_per_second', 'throughput', 'rate', 'bps'],
                'cic_variants': ['Flow Bytes/s', 'flow bytes/s', 'Flow Bytes per second'],
                'category': 'rate',
                'unit': 'bytes_per_second'
            },
            # Flow Rate - Packets per second
            {
                'unified_name': 'flow_packets_per_sec',
                'ton_variants': ['flow_packets_per_sec', 'packets_per_second', 'pps'],
                'cic_variants': ['Flow Packets/s', 'flow packets/s', 'Flow Packets per second'],
                'category': 'rate',
                'unit': 'packets_per_second'
            },
            # Flags - SYN
            {
                'unified_name': 'syn_flags',
                'ton_variants': ['syn', 'syn_flag', 'syn_count', 'syn_flags'],
                'cic_variants': ['SYN Flag Count', 'syn flag count', 'SYN'],
                'category': 'flag',
                'unit': 'count'
            },
            # Flags - ACK
            {
                'unified_name': 'ack_flags',
                'ton_variants': ['ack', 'ack_flag', 'ack_count', 'ack_flags'],
                'cic_variants': ['ACK Flag Count', 'ack flag count', 'ACK'],
                'category': 'flag',
                'unit': 'count'
            },
            # Flags - FIN
            {
                'unified_name': 'fin_flags',
                'ton_variants': ['fin', 'fin_flag', 'fin_count', 'fin_flags'],
                'cic_variants': ['FIN Flag Count', 'fin flag count', 'FIN'],
                'category': 'flag',
                'unit': 'count'
            },
            # IAT (Inter-Arrival Time) - Forward Mean
            {
                'unified_name': 'fwd_iat_mean',
                'ton_variants': ['fwd_iat', 'forward_iat', 'iat_mean', 'fwd_iat_mean'],
                'cic_variants': ['Fwd IAT Mean', 'fwd iat mean', 'Forward IAT Mean'],
                'category': 'temporal',
                'unit': 'microseconds'
            },
            # IAT - Backward Mean
            {
                'unified_name': 'bwd_iat_mean',
                'ton_variants': ['bwd_iat', 'backward_iat', 'bwd_iat_mean'],
                'cic_variants': ['Bwd IAT Mean', 'bwd iat mean', 'Backward IAT Mean'],
                'category': 'temporal',
                'unit': 'microseconds'
            },
            # Packet Length - Forward Mean
            {
                'unified_name': 'fwd_packet_length_mean',
                'ton_variants': ['fwd_packet_length', 'fwd_pkt_len', 'forward_packet_length', 'packet_length'],
                'cic_variants': ['Fwd Packet Length Mean', 'fwd packet length mean'],
                'category': 'byte',
                'unit': 'bytes'
            },
            # Packet Length - Backward Mean
            {
                'unified_name': 'bwd_packet_length_mean',
                'ton_variants': ['bwd_packet_length', 'bwd_pkt_len', 'backward_packet_length'],
                'cic_variants': ['Bwd Packet Length Mean', 'bwd packet length mean'],
                'category': 'byte',
                'unit': 'bytes'
            },
        ]
        
        for mapping in known_mappings:
            ton_match = None
            cic_match = None
            
            # Find TON match
            for variant in mapping['ton_variants']:
                for col in df_ton.columns:
                    if variant.lower() in col.lower():
                        ton_match = col
                        break
                if ton_match:
                    break
            
            # Find CIC match
            for variant in mapping['cic_variants']:
                for col in df_cic.columns:
                    if variant.lower() in col.lower():
                        cic_match = col
                        break
                if cic_match:
                    break
            
            if ton_match and cic_match:
                mappings.append({
                    'unified_name': mapping['unified_name'],
                    'ton_name': ton_match,
                    'cic_name': cic_match,
                    'type': 'known_mapping',
                    'category': mapping['category'],
                    'unit': mapping['unit'],
                    'confidence': 'high'
                })
        
        return mappings
    
    def generate_feature_mapping_report(self, output_path: Optional[Path] = None) -> str:
        """Generate a detailed report of feature mappings"""
        if not self.common_features_proposed:
            return "No common features found. Run extract_common_features() first."
        
        report = "# Feature Mapping Report: TON_IoT ↔ CIC-DDoS2019\n\n"
        report += f"**Total Common Features Found**: {len(self.common_features_proposed)}\n\n"
        
        # Group by type
        by_type = {}
        for feat in self.common_features_proposed:
            feat_type = feat['type']
            if feat_type not in by_type:
                by_type[feat_type] = []
            by_type[feat_type].append(feat)
        
        for feat_type, features in by_type.items():
            report += f"## {feat_type.replace('_', ' ').title()}\n\n"
            report += f"**Count**: {len(features)}\n\n"
            report += "| Unified Name | TON_IoT | CIC-DDoS2019 | Category | Unit | Confidence |\n"
            report += "|--------------|---------|---------------|----------|------|------------|\n"
            
            for feat in sorted(features, key=lambda x: x.get('similarity_score', 1.0), reverse=True):
                unified = feat['unified_name']
                ton = feat['ton_name']
                cic = feat['cic_name']
                cat = feat.get('category', 'N/A')
                unit = feat.get('unit', 'N/A')
                conf = feat.get('confidence', 'N/A')
                score = feat.get('similarity_score', 'N/A')
                
                report += f"| `{unified}` | `{ton}` | `{cic}` | {cat} | {unit} | {conf} |\n"
            
            report += "\n"
        
        # Summary by category
        report += "## Summary by Category\n\n"
        by_category = {}
        for feat in self.common_features_proposed:
            cat = feat.get('category', 'other')
            if cat not in by_category:
                by_category[cat] = 0
            by_category[cat] += 1
        
        for cat, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
            report += f"- **{cat}**: {count} features\n"
        
        if output_path:
            output_path.write_text(report)
            logger.info(f"Feature mapping report saved to {output_path}")
        
        return report


def analyze_and_propose_features(ton_path: Optional[Path] = None, cic_path: Optional[Path] = None) -> List[Dict]:
    """
    Analyze datasets and propose common features
    
    Args:
        ton_path: Path to TON_IoT CSV (optional, will search if None)
        cic_path: Path to CIC-DDoS2019 CSV or directory (optional, will search if None)
        
    Returns:
        List of proposed common feature mappings
    """
    from dataset_loader import DatasetLoader
    
    loader = DatasetLoader()
    analyzer = FeatureAnalyzer()
    
    # Load datasets (small sample for analysis)
    if ton_path is None:
        df_ton = loader.load_ton_iot(sample_ratio=0.01)  # 1% for analysis
    else:
        df_ton = pd.read_csv(ton_path, nrows=1000)
    
    if cic_path is None:
        df_cic = loader.load_cic_ddos2019(sample_ratio=0.01)  # 1% for analysis
    else:
        if Path(cic_path).is_dir():
            df_cic = loader.load_cic_ddos2019(dataset_path=cic_path, sample_ratio=0.01)
        else:
            df_cic = pd.read_csv(cic_path, nrows=1000)
    
    # Extract common features
    common_features = analyzer.extract_common_features(df_ton, df_cic)
    
    # Generate report
    report = analyzer.generate_feature_mapping_report()
    print(report)
    
    return common_features


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    analyze_and_propose_features()
