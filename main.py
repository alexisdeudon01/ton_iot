#!/usr/bin/env python3
"""
Main entry point for IRP Research Pipeline
AI-Powered Log Analysis for Smarter Threat Detection

Usage:
    python main.py                    # Run complete pipeline (all phases)
    python main.py --phase 1          # Run only Phase 1 (config search)
    python main.py --phase 2          # Run only Phase 2 (apply best config)
    python main.py --phase 3          # Run only Phase 3 (evaluation)
    python main.py --phase 4          # Run only Phase 4 (AHP preferences)
    python main.py --phase 5          # Run only Phase 5 (TOPSIS ranking)
    python main.py --output-dir custom_output  # Custom output directory
    python main.py --interactive      # Enable Tkinter UI (optional)
    python main.py --test-mode        # Run in test mode (0.001% data)
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import os

# Fonction pour vérifier la disponibilité du GPU
def check_gpu_availability():
    """Check GPU availability (optional, requires torch)"""
    try:
        import torch
        is_gpu_available = torch.cuda.is_available()
        return is_gpu_available
    except ImportError:
        # torch not installed, assume CPU only
        return False

# Fonction pour générer le fichier req2.txt basé sur le matériel
def generate_requirements(is_gpu_available):
    base_requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "imbalanced-learn>=0.10.0",
        "shap>=0.41.0",
        "lime>=0.2.0",
        "psutil>=5.9.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.64.0"
    ]

    if is_gpu_available:
        # Dépendances pour GPU
        additional_requirements = [
            "torch>=1.12.0+cu113",  # PyTorch avec CUDA 11.3
            "pytorch-tabnet>=4.0",  # TabNet
            "xgboost>=1.5.2"  # XGBoost avec support CUDA
        ]
    else:
        # Dépendances pour CPU seulement
        additional_requirements = [
            "torch>=1.12.0",  # PyTorch CPU
            "pytorch-tabnet>=4.0",  # TabNet CPU
            "xgboost<3"  # XGBoost sans CUDA
        ]

    # Fusionner les dépendances de base avec les dépendances spécifiques à la machine
    all_requirements = base_requirements + additional_requirements

    # Sauvegarder dans un fichier req2.txt
    with open("req2.txt", "w") as f:
        for req in all_requirements:
            f.write(req + "\n")

    print(f"Le fichier 'req2.txt' a été généré avec les dépendances adaptées.")

# Fonction pour configurer le logging
def setup_logging(output_dir: Path):
    """Setup logging configuration"""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'main_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

# Fonction pour vérifier les dépendances nécessaires
def check_requirements():
    """Check if required packages are installed"""
    try:
        import pandas
        import numpy
        import sklearn
        import tqdm
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e.name}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

# Fonction pour demander si on est en mode test
def ask_test_mode() -> bool:
    """Ask user for test mode (only if interactive)"""
    if not sys.stdout.isatty():
        return False

    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()

        response = messagebox.askyesno(
            "Test Mode",
            "Do you want to run in TEST MODE?\n\n"
            "YES: Use 0.001% of data (faster, for testing)\n"
            "NO: Use 100% of data (slower, full pipeline)"
        )
        root.destroy()
        return response
    except Exception:
        # Tkinter not available, default to False
        return False

# Fonction principale
def main():
    """Main entry point"""
    # Ajouter le chemin src
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    # Vérifier les dépendances
    if not check_requirements():
        sys.exit(1)

    # Vérification GPU et génération du fichier req2.txt
    is_gpu_available = check_gpu_availability()
    generate_requirements(is_gpu_available)

    # Parser les arguments CLI
    from src.app.cli import parse_args, args_to_config

    args = parse_args()
    config = args_to_config(args)

    # Setup logging
    logger = setup_logging(Path(config.output_dir))

    # Demander si on veut être en mode test
    if not args.test_mode and not args.sample_ratio and config.interactive:
        test_mode = ask_test_mode()
        if test_mode:
            config.test_mode = True
            config.sample_ratio = 0.001
            logger.info("=" * 70)
            logger.info("⚠️  MODE TEST ACTIVÉ - Utilisation de 0.001% des données")
            logger.info("=" * 70)

    # Initialiser et exécuter le pipeline
    try:
        # Use the new pipeline V7 as requested
        from src.new_pipeline.main import main as run_new_pipeline

        logger.info("=" * 70)
        logger.info("IRP RESEARCH PIPELINE V7 - NEW ARCHITECTURE")
        logger.info("AI-Powered Log Analysis for Smarter Threat Detection")
        logger.info("=" * 70)

        # Run the new pipeline
        run_new_pipeline()

        logger.info("=" * 70)
        logger.info("✅ PIPELINE EXECUTION SUCCESSFUL")
        logger.info("=" * 70)

        return 0

    except KeyboardInterrupt:
        logger.info("\n⚠️ Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Error in pipeline execution: {e}", exc_info=True)
        logger.error("See log file for full traceback")
        return 1


if __name__ == "__main__":
    sys.exit(main())
