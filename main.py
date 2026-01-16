#!/usr/bin/env python3
"""
Main entry point for IRP Research Pipeline
AI-Powered Log Analysis for Smarter Threat Detection

Usage:
    python main.py                    # Run complete pipeline
    python main.py --phase 1          # Run only Phase 1
    python main.py --phase 3          # Run only Phase 3
    python main.py --phase 5          # Run only Phase 5
    python main.py --output-dir custom_output  # Custom output directory
"""

import sys
import argparse
import logging
import subprocess
import re
from pathlib import Path
from datetime import datetime

# Setup logging first (before importing pipeline)
def setup_logging(output_dir: Path):
    """Setup logging configuration"""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'main_{timestamp}.log'
    
    # Configure logging
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


def check_requirements(output_dir: Path):
    """
    Check if all required packages from requirements.txt are installed
    
    Args:
        output_dir: Output directory for logging
        
    Returns:
        True if all requirements are met, False otherwise
    """
    logger = logging.getLogger(__name__)
    requirements_file = Path('requirements.txt')
    
    if not requirements_file.exists():
        logger.warning("requirements.txt not found. Skipping dependency check.")
        return True
    
    logger.info("Checking Python dependencies from requirements.txt...")
    
    try:
        with open(requirements_file, 'r') as f:
            requirements = f.readlines()
        
        missing_packages = []
        installed_packages = {}
        
        # Get installed packages
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--format=freeze'],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if '==' in line:
                        pkg_name = line.split('==')[0].lower().replace('-', '_').replace('.', '_')
                        installed_packages[pkg_name] = line
        except Exception as e:
            logger.warning(f"Could not check installed packages: {e}")
            return True  # Continue anyway
        
        # Parse requirements.txt
        for req_line in requirements:
            req_line = req_line.strip()
            # Skip comments and empty lines
            if not req_line or req_line.startswith('#'):
                continue
            
            # Extract package name (handle version constraints)
            match = re.match(r'^([a-zA-Z0-9_-]+(?:\[.*\])?)', req_line)
            if match:
                pkg_name_raw = match.group(1)
                # Remove extras [optional]
                pkg_name_clean = re.sub(r'\[.*\]', '', pkg_name_raw)
                pkg_name_normalized = pkg_name_clean.lower().replace('-', '_').replace('.', '_')
                
                # Check if package is installed
                found = False
                for installed_name in installed_packages.keys():
                    if installed_name.startswith(pkg_name_normalized) or pkg_name_normalized.startswith(installed_name):
                        found = True
                        break
                
                if not found:
                    missing_packages.append(pkg_name_clean)
        
        if missing_packages:
            logger.warning(f"Missing packages ({len(missing_packages)}): {', '.join(missing_packages[:5])}")
            if len(missing_packages) > 5:
                logger.warning(f"... and {len(missing_packages) - 5} more")
            logger.info("Install missing packages with: pip install -r requirements.txt")
            return False
        else:
            logger.info("All required packages are installed.")
            return True
            
    except Exception as e:
        logger.warning(f"Error checking requirements: {e}")
        return True  # Continue anyway


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='IRP Research Pipeline: AI-Powered Log Analysis for Smarter Threat Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Run complete pipeline
  python main.py --phase 1                    # Run only Phase 1 (preprocessing)
  python main.py --phase 3                    # Run only Phase 3 (evaluation)
  python main.py --phase 5                    # Run only Phase 5 (ranking)
  python main.py --output-dir custom_output   # Use custom output directory
        """
    )
    
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 3, 5],
        help='Run specific phase only (1: preprocessing, 3: evaluation, 5: ranking)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for results (default: output)'
    )
    
    args = parser.parse_args()
    
    # Setup output directory and logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    
    # Welcome message
    logger.info("=" * 70)
    logger.info("IRP RESEARCH PIPELINE")
    logger.info("AI-Powered Log Analysis for Smarter Threat Detection")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check requirements
    if not check_requirements(output_dir):
        logger.warning("Some dependencies may be missing. Continue anyway? (y/n)")
        # For non-interactive mode, continue anyway but log warning
        logger.warning("Continuing execution despite missing dependencies...")
    
    if args.phase:
        logger.info(f"Running Phase {args.phase} only")
    else:
        logger.info("Running complete pipeline (Phases 1, 3, 5)")
    
    try:
        # Import pipeline (after logging setup)
        # Add src directory to Python path
        src_path = Path(__file__).parent / 'src'
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        from main_pipeline import IRPPipeline
        
        # Initialize pipeline
        pipeline = IRPPipeline(results_dir=str(output_dir), random_state=42)
        
        if args.phase == 1:
            # Phase 1 only
            logger.info("Starting Phase 1: Preprocessing Configuration Selection")
            X, y, feature_names = pipeline.phase1_preprocessing()
            logger.info("Phase 1 completed successfully")
            
        elif args.phase == 3:
            # Phase 3 only (requires Phase 1 output)
            logger.warning("Phase 3 requires preprocessed data from Phase 1.")
            logger.warning("Attempting to load from output directory...")
            try:
                import pandas as pd
                import numpy as np
                preprocessed = pd.read_csv(output_dir / 'phase1_preprocessing' / 'preprocessed_data.csv')
                y = preprocessed['label'].values
                X_df = preprocessed.drop('label', axis=1)
                X = X_df.values
                feature_names = list(X_df.columns)
                logger.info(f"Loaded preprocessed data: {X.shape}")
            except FileNotFoundError:
                logger.error("Preprocessed data not found. Please run Phase 1 first.")
                sys.exit(1)
            
            logger.info("Starting Phase 3: Multi-Dimensional Algorithm Evaluation")
            evaluation_results = pipeline.phase3_evaluation(X, y, feature_names)
            logger.info("Phase 3 completed successfully")
            
        elif args.phase == 5:
            # Phase 5 only (requires Phase 3 output)
            logger.warning("Phase 5 requires evaluation results from Phase 3.")
            logger.warning("Attempting to load from output directory...")
            try:
                import pandas as pd
                evaluation_results = pd.read_csv(output_dir / 'phase3_evaluation' / 'evaluation_results.csv')
                logger.info(f"Loaded evaluation results: {len(evaluation_results)} algorithms")
            except FileNotFoundError:
                logger.error("Evaluation results not found. Please run Phase 3 first.")
                sys.exit(1)
            
            logger.info("Starting Phase 5: AHP-TOPSIS Ranking")
            ranking_results = pipeline.phase5_ranking(evaluation_results)
            logger.info("Phase 5 completed successfully")
            
        else:
            # Run complete pipeline
            logger.info("Starting complete pipeline execution...")
            pipeline.run()
            logger.info("Pipeline completed successfully")
        
        logger.info("=" * 70)
        logger.info("EXECUTION COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Results available in: {output_dir.absolute()}")
        logger.info("  - Phase 1: phase1_preprocessing/")
        logger.info("  - Phase 3: phase3_evaluation/")
        logger.info("  - Phase 5: phase5_ranking/")
        logger.info(f"  - Logs: logs/")
        logger.info("")
        logger.info("To visualize results with GUI:")
        logger.info(f"  python -m src.results_visualizer --output-dir {output_dir}")
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (Ctrl+C)")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"Error in pipeline execution: {e}", exc_info=True)
        logger.error("See log file for full traceback")
        sys.exit(1)


if __name__ == "__main__":
    main()
