#!/usr/bin/env python3
"""
Script to analyze TON_IoT and CIC-DDoS2019 CSV files and propose common features
Examines actual columns, units, and data types
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import logging
import pandas as pd
from feature_analyzer import FeatureAnalyzer
from dataset_loader import DatasetLoader

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function to analyze features"""
    print("=" * 70)
    print("ANALYSE DES FEATURES COMMUNES: TON_IoT ↔ CIC-DDoS2019")
    print("=" * 70)
    print()
    
    loader = DatasetLoader()
    analyzer = FeatureAnalyzer()
    
    # Load small samples for analysis
    print("📁 Chargement des datasets (échantillon 1% pour analyse)...")
    print()
    
    try:
        df_ton = loader.load_ton_iot(sample_ratio=0.01)
        print(f"✅ TON_IoT chargé: {df_ton.shape[0]:,} lignes, {df_ton.shape[1]} colonnes")
        print(f"   Colonnes: {', '.join(df_ton.columns[:10].tolist())}...")
        print()
    except Exception as e:
        logger.error(f"Erreur chargement TON_IoT: {e}")
        return
    
    try:
        df_cic = loader.load_cic_ddos2019(sample_ratio=0.01)
        print(f"✅ CIC-DDoS2019 chargé: {df_cic.shape[0]:,} lignes, {df_cic.shape[1]} colonnes")
        print(f"   Colonnes: {', '.join(df_cic.columns[:10].tolist())}...")
        print()
    except Exception as e:
        logger.error(f"Erreur chargement CIC-DDoS2019: {e}")
        return
    
    # Analyze features
    print("🔍 Analyse des features...")
    print()
    
    common_features = analyzer.extract_common_features(df_ton, df_cic)
    
    # Generate report
    output_path = Path('output') / 'feature_mapping_report.md'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = analyzer.generate_feature_mapping_report(output_path)
    
    print("=" * 70)
    print("RAPPORT GÉNÉRÉ")
    print("=" * 70)
    print()
    print(report)
    print()
    print(f"📄 Rapport complet sauvegardé dans: {output_path}")
    print()
    
    # Summary
    print("=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"Total features communes trouvées: {len(common_features)}")
    
    by_type = {}
    by_category = {}
    for feat in common_features:
        feat_type = feat['type']
        by_type[feat_type] = by_type.get(feat_type, 0) + 1
        
        cat = feat.get('category', 'other')
        by_category[cat] = by_category.get(cat, 0) + 1
    
    print("\nPar type:")
    for feat_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {feat_type}: {count}")
    
    print("\nPar catégorie:")
    for cat, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {cat}: {count}")
    
    print()
    print("✅ Analyse terminée!")


if __name__ == "__main__":
    main()
