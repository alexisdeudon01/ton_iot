#!/usr/bin/env python3
"""
Main entry point for IRP Research Pipeline - LATE FUSION EDITION
AI-Powered Log Analysis for Smarter Threat Detection
"""

import sys 
import logging
import argparse
from pathlib import Path
from datetime import datetime
import os

# Configuration du chemin src
sys.path.insert(0, str(Path(__file__).parent))

def check_dependencies():
    """Vérifie les dépendances critiques avant le lancement."""
    import importlib
    deps = ["dask", "pandas", "numpy", "sklearn", "torch", "shap", "joblib", "pydantic", "pyarrow"]
    missing = []
    for dep in deps:
        try:
            importlib.import_module(dep)
        except ImportError:
            missing.append(dep)
    
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    return True

def setup_logging(output_dir: Path):
    """Configure le logging global."""
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
    return logging.getLogger(__name__)

def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="IRP Research Pipeline - Late Fusion")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode (0.1% data)")
    parser.add_argument("--sample-ratio", type=float, default=1.0, help="Data sampling ratio (0.0 to 1.0)")
    parser.add_argument("--force-prepare", action="store_true", help="Force CSV to Parquet conversion")
    parser.add_argument("--output-dir", type=str, default="output", help="Custom output directory")
    return parser.parse_args()

def main():
    """Point d'entrée principal unifié."""
    args = parse_args()
    
    if not check_dependencies():
        sys.exit(1)

    # Initialisation de la configuration Pydantic
    from src.new_pipeline.config import config
    config.test_mode = args.test_mode
    config.paths.out_root = Path(args.output_dir)
    
    # Setup logging
    logger = setup_logging(config.paths.out_root)
    
    logger.info("=" * 70)
    logger.info("IRP RESEARCH PIPELINE - LATE FUSION ARCHITECTURE")
    logger.info(f"Mode: {'TEST (0.1%)' if config.test_mode else 'NORMAL'}")
    logger.info("=" * 70)

    try:
        # Import et exécution du pipeline V8 (Late Fusion)
        from src.new_pipeline.main import main as run_pipeline
        
        # On injecte les arguments CLI dans l'exécution
        # Note: run_pipeline() dans src/new_pipeline/main.py utilise l'objet config global
        run_pipeline()

        logger.info("=" * 70)
        logger.info("✅ PIPELINE EXECUTION SUCCESSFUL")
        logger.info("=" * 70)
        return 0

    except KeyboardInterrupt:
        logger.info("\n⚠️ Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Error in pipeline execution: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
