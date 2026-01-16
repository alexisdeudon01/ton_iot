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

# Setup logging first
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


def ask_test_mode() -> bool:
    """Ask user for test mode (only if interactive)"""
    # Only ask if we're in a terminal (not in CI/automated run)
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


def main():
    """Main entry point"""
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Parse CLI arguments
    from src.app.cli import parse_args, args_to_config
    
    args = parse_args()
    config = args_to_config(args)
    
    # Setup logging
    logger = setup_logging(Path(config.output_dir))
    
    # Ask for test mode if not specified and interactive
    if not args.test_mode and not args.sample_ratio and config.interactive:
        test_mode = ask_test_mode()
        if test_mode:
            config.test_mode = True
            config.sample_ratio = 0.001
            logger.info("=" * 70)
            logger.info("⚠️  MODE TEST ACTIVÉ - Utilisation de 0.001% des données")
            logger.info("=" * 70)
    
    # Initialize and run pipeline
    try:
        from src.app.pipeline_runner import PipelineRunner
        
        logger.info("=" * 70)
        logger.info("IRP RESEARCH PIPELINE")
        logger.info("AI-Powered Log Analysis for Smarter Threat Detection")
        logger.info("=" * 70)
        logger.info(f"Output directory: {config.output_dir}")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Determine phases to run
        phases = [args.phase] if args.phase else None
        
        # Create runner and execute
        runner = PipelineRunner(config)
        results = runner.run(phases=phases)
        
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
