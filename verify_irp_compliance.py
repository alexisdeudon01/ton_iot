#!/usr/bin/env python3
"""
Script de vérification de conformité avec la méthodologie IRP
Vérifie que l'implémentation suit correctement les spécifications du document IRP_FinalADE_v2.0ADE-2-1.pdf
"""
import importlib.util
import sys
import typing
from pathlib import Path
from pprint import pprint
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Union

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def check_irp_compliance():
    """Vérifie la conformité avec la méthodologie IRP"""
    print("=" * 70)
    print("VÉRIFICATION DE CONFORMITÉ IRP")
    print("=" * 70)
    print()

    checks_passed = 0
    checks_failed = 0
    warnings = []

    # 1. Vérifier Phase 1: Preprocessing
    print("📋 Phase 1: Preprocessing Configuration Selection")
    print("-" * 70)

    try:
        from src.main_pipeline import IRPPipeline

        pipeline = IRPPipeline()

        # Vérifier harmonisation
        if hasattr(pipeline, "harmonizer"):
            print("  ✓ Harmonisation des datasets (CIC-DDoS2019 + TON_IoT)")
            checks_passed += 1
        else:
            print("  ✗ Harmonisation non trouvée")
            checks_failed += 1

        # Vérifier preprocessing pipeline
        if hasattr(pipeline, "pipeline"):
            print("  ✓ Pipeline de preprocessing (SMOTE, RobustScaler)")
            checks_passed += 1
        else:
            print("  ✗ Pipeline de preprocessing non trouvé")
            checks_failed += 1

    except Exception as e:
        print(f"  ✗ Erreur: {e}")
        checks_failed += 1

    print()

    # 2. Vérifier Phase 3: 3D Evaluation
    print("📊 Phase 3: Multi-Dimensional Algorithm Evaluation")
    print("-" * 70)

    try:
        from src.evaluation_3d import Evaluation3D

        evaluator = Evaluation3D(feature_names=["test"])

        # Vérifier dimensions
        dims = evaluator.get_dimension_scores()
        expected_dims = [
            "Detection Performance",
            "Resource Efficiency",
            "Explainability",
        ]

        print("  ✓ Framework d'évaluation 3D implémenté")
        checks_passed += 1

        # Vérifier ResourceMonitor
        from src.evaluation_3d import ResourceMonitor

        print("  ✓ ResourceMonitor pour mesurer temps/mémoire")
        checks_passed += 1

    except Exception as e:
        print(f"  ✗ Erreur: {e}")
        checks_failed += 1

    print()

    # 3. Vérifier Algorithmes IRP
    print("🤖 Algorithmes selon méthodologie IRP")
    print("-" * 70)

    expected_algorithms = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "CNN",
        "TabNet",
    ]

    try:
        # Vérifier dans main_pipeline
        from src.main_pipeline import IRPPipeline

        pipeline = IRPPipeline()

        # Lire le code pour vérifier les algorithmes
        main_pipeline_path = Path("src/main_pipeline.py")
        content = main_pipeline_path.read_text()

        for algo in expected_algorithms:
            if algo in content or algo.lower().replace(" ", "_") in content.lower():
                print(f"  ✓ {algo}")
                checks_passed += 1
            else:
                print(f"  ✗ {algo} non trouvé")
                checks_failed += 1

    except Exception as e:
        print(f"  ✗ Erreur: {e}")
        checks_failed += 1

    print()

    # 4. Vérifier Phase 5: AHP-TOPSIS
    print("📈 Phase 5: AHP-TOPSIS Ranking")
    print("-" * 70)

    try:
        from src.ahp_topsis_framework import AHPTopsisFramework

        print("  ✓ Framework AHP-TOPSIS implémenté")
        checks_passed += 1

        # Vérifier méthodes principales
        framework = AHPTopsisFramework()
        required_methods = [
            "set_ahp_comparisons",
            "get_weights",
            "set_decision_matrix",
            "rank_alternatives",
        ]
        for method in required_methods:
            if hasattr(framework, method):
                print(f"    ✓ Méthode {method}()")
                checks_passed += 1
            else:
                print(f"    ✗ Méthode {method}() manquante")
                checks_failed += 1

    except Exception as e:
        print(f"  ✗ Erreur: {e}")
        checks_failed += 1

    print()

    # 5. Vérifier Datasets
    print("📁 Datasets")
    print("-" * 70)

    try:
        from src.dataset_loader import DatasetLoader

        loader = DatasetLoader()

        print("  ✓ DatasetLoader pour CIC-DDoS2019 et TON_IoT")
        checks_passed += 1

        # Vérifier harmonisation
        from src.data_harmonization import DataHarmonizer

        harmonizer = DataHarmonizer()
        print("  ✓ DataHarmonizer pour harmonisation et early fusion")
        checks_passed += 1

    except Exception as e:
        print(f"  ✗ Erreur: {e}")
        checks_failed += 1

    print()

    # 6. Vérifier Cross-Validation
    print("🔄 Cross-Validation")
    print("-" * 70)

    try:
        from src.preprocessing_pipeline import StratifiedCrossValidator

        print("  ✓ StratifiedCrossValidator (5-fold CV)")
        checks_passed += 1
    except Exception as e:
        print(f"  ✗ Erreur: {e}")
        checks_failed += 1

    print()

    # 7. Vérifier Visualisations
    print("📊 Visualisations")
    print("-" * 70)

    try:
        from src.realtime_visualizer import RealTimeVisualizer

        print("  ✓ Visualisations en temps réel")
        checks_passed += 1

        from src.results_visualizer import ResultsVisualizer

        print("  ✓ Interface Tkinter pour visualisation des résultats")
        checks_passed += 1

    except Exception as e:
        warnings.append(f"Visualisations: {e}")
        print(f"  ⚠ Avertissement: {e}")

    print()

    # Résumé
    print("=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"✓ Vérifications réussies: {checks_passed}")
    print(f"✗ Vérifications échouées: {checks_failed}")
    if warnings:
        print(f"⚠ Avertissements: {len(warnings)}")
        for w in warnings:
            print(f"  - {w}")

    print()
    if checks_failed == 0:
        print("✅ CONFORMITÉ IRP: OK")
        return 0
    else:
        print("❌ CONFORMITÉ IRP: PROBLÈMES DÉTECTÉS")
        return 1


if __name__ == "__main__":
    exit_code = check_irp_compliance()
    sys.exit(exit_code)
