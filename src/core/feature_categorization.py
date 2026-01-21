import pandas as pd
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

FEATURE_CATEGORIES = {
    'Flow_Identifiers': [
        'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp',
        'src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto', 'ts'
    ],
    'Flow_Basic_Stats': [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets',
        'Total Length of Bwd Packets', 'duration', 'pkts_in', 'pkts_out', 'bytes_in', 'bytes_out'
    ],
    'Packet_Length_Stats': [
        'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
        'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
        'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
        'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size'
    ],
    'IAT_Stats': [
        'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
        'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
        'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min'
    ],
    'Flag_Counts': [
        'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
        'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
        'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count'
    ],
    'Header_Stats': [
        'Fwd Header Length', 'Bwd Header Length', 'Fwd Header Length.1'
    ],
    'Throughput_Stats': [
        'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packets/s', 'Bwd Packets/s'
    ],
    'Bulk_Stats': [
        'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
        'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate'
    ],
    'Subflow_Stats': [
        'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes'
    ],
    'Window_Segment_Stats': [
        'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward'
    ],
    'Active_Idle_Stats': [
        'Active Mean', 'Active Std', 'Active Max', 'Active Min',
        'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
    ],
    'Other': [
        'SimillarHTTP', 'Inbound'
    ]
}

# Mapping categories to measurement criteria (Performance, Explainability, Resource Consumption)
# Scores from 1 to 10
CATEGORY_METRICS = {
    'Flow_Identifiers': {'performance': 2, 'explainability': 10, 'resources': 1},
    'Flow_Basic_Stats': {'performance': 5, 'explainability': 9, 'resources': 2},
    'Packet_Length_Stats': {'performance': 8, 'explainability': 7, 'resources': 4},
    'IAT_Stats': {'performance': 9, 'explainability': 5, 'resources': 8},
    'Flag_Counts': {'performance': 4, 'explainability': 10, 'resources': 1},
    'Header_Stats': {'performance': 3, 'explainability': 8, 'resources': 2},
    'Throughput_Stats': {'performance': 7, 'explainability': 6, 'resources': 5},
    'Bulk_Stats': {'performance': 6, 'explainability': 4, 'resources': 7},
    'Subflow_Stats': {'performance': 5, 'explainability': 7, 'resources': 3},
    'Window_Segment_Stats': {'performance': 6, 'explainability': 6, 'resources': 4},
    'Active_Idle_Stats': {'performance': 8, 'explainability': 5, 'resources': 6},
    'Other': {'performance': 1, 'explainability': 3, 'resources': 1}
}

def categorize_features(columns: List[str]) -> Dict[str, List[str]]:
    """Categorizes features into common groups."""
    categorized = {cat: [] for cat in FEATURE_CATEGORIES.keys()}

    for col in columns:
        found = False
        for cat, features in FEATURE_CATEGORIES.items():
            if col in features:
                categorized[cat].append(col)
                found = True
                break
        if not found:
            categorized['Other'].append(col)

    return categorized

def get_category_scores(categorized_features: Dict[str, List[str]]) -> Dict[str, float]:
    """Calculates average scores for each criterion based on present features."""
    scores = {'performance': 0.0, 'explainability': 0.0, 'resources': 0.0}
    total_features = sum(len(f) for f in categorized_features.values())

    if total_features == 0:
        return scores

    for cat, features in categorized_features.items():
        weight = len(features) / total_features
        for criterion in scores.keys():
            scores[criterion] += CATEGORY_METRICS[cat][criterion] * weight

    return scores

def print_verbose_feature_info(categorized_features: Dict[str, List[str]], normalization_method: str = "RobustScaler"):
    """Prints detailed information about feature categories and normalization."""
    print("\n" + "="*80)
    print("PROMPT EXPERT: ANALYSE DES CATÉGORIES DE FEATURES")
    print("="*80)

    active_categories = {k: v for k, v in categorized_features.items() if len(v) > 0}

    print(f"Nombre de catégories identifiées : {len(active_categories)}")
    if len(active_categories) < 4:
        print("⚠️ ALERTE : Moins de 4 catégories identifiées. Vérifiez le mapping des features.")

    print(f"\nMéthode de normalisation utilisée : {normalization_method}")
    print("Raison : Gestion robuste des valeurs aberrantes (outliers) fréquente dans le trafic réseau.")

    for cat, features in active_categories.items():
        print(f"\n📂 CATÉGORIE : {cat}")
        print(f"   Description : {cat.replace('_', ' ')}")
        print(f"   Features ({len(features)}) : {', '.join(features[:10])}{'...' if len(features) > 10 else ''}")
        metrics = CATEGORY_METRICS.get(cat, {})
        print(f"   Scores (1-10) -> Perf: {metrics.get('performance')}, XAI: {metrics.get('explainability')}, Res: {metrics.get('resources')}")

    print("="*80 + "\n")
