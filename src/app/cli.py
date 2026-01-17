#!/usr/bin/env python3
"""
CLI parsing pour le pipeline IRP
Sans dépendance UI (Tkinter optionnel via --interactive)
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

# Import from config module (src/config.py, not src/config/ package)
_parent_dir = Path(__file__).parent.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from src.config import PipelineConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='IRP Research Pipeline - AI-Powered Log Analysis for Smarter Threat Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run complete pipeline (all phases)
  python main.py --phase 1          # Run only Phase 1 (config search)
  python main.py --phase 2          # Run only Phase 2 (apply best config)
  python main.py --phase 3          # Run only Phase 3 (evaluation)
  python main.py --phase 4          # Run only Phase 4 (AHP preferences)
  python main.py --phase 5          # Run only Phase 5 (TOPSIS ranking)
  python main.py --output-dir custom_output  # Custom output directory
  python main.py --interactive      # Enable Tkinter UI (optional)
  python main.py --test-mode        # Run in test mode (0.1% data)
        """
    )
    
    # Phase selection
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Run only a specific phase (1-5). If not specified, runs all phases.'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for results (default: output)'
    )
    
    # Mode
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode with minimal data (sample_ratio=0.001)'
    )
    
    parser.add_argument(
        '--sample-ratio',
        type=float,
        default=None,
        help='Ratio of data to use (0.0-1.0). Overrides --test-mode if specified.'
    )
    
    parser.add_argument(
        '--cic-max-files',
        type=int,
        default=None,
        help='Maximum number of CIC-DDoS2019 files to load (default: 3 in test mode, all in normal mode)'
    )
    
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic dataset for Phase 3 evaluation (generated via sklearn.datasets.make_classification)'
    )
    
    # UI
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Enable interactive Tkinter UI (optional, requires GUI environment)'
    )
    
    # AHP Preferences (Phase 4)
    parser.add_argument(
        '--ahp-dim1-weight',
        type=float,
        default=None,
        help='AHP weight for Dimension 1 (Performance). Must sum to 1.0 with dim2 and dim3.'
    )
    parser.add_argument(
        '--ahp-dim2-weight',
        type=float,
        default=None,
        help='AHP weight for Dimension 2 (Resources). Must sum to 1.0 with dim1 and dim3.'
    )
    parser.add_argument(
        '--ahp-dim3-weight',
        type=float,
        default=None,
        help='AHP weight for Dimension 3 (Explainability). Must sum to 1.0 with dim1 and dim2.'
    )
    
    # Random state
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> PipelineConfig:
    """
    Convert CLI arguments to PipelineConfig
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        PipelineConfig instance
    """
    # Determine sample ratio
    sample_ratio = 0.001 if args.test_mode else (args.sample_ratio if args.sample_ratio is not None else 1.0)
    
    # AHP preferences
    ahp_preferences = {
        'dimension1_performance': 0.5,
        'dimension2_resources': 0.3,
        'dimension3_explainability': 0.2
    }
    
    # Override with CLI if provided
    if args.ahp_dim1_weight is not None or args.ahp_dim2_weight is not None or args.ahp_dim3_weight is not None:
        dim1 = args.ahp_dim1_weight if args.ahp_dim1_weight is not None else 0.5
        dim2 = args.ahp_dim2_weight if args.ahp_dim2_weight is not None else 0.3
        dim3 = args.ahp_dim3_weight if args.ahp_dim3_weight is not None else 0.2
        
        # Normalize to sum to 1.0
        total = dim1 + dim2 + dim3
        if total > 0:
            dim1 /= total
            dim2 /= total
            dim3 /= total
        
        ahp_preferences = {
            'dimension1_performance': dim1,
            'dimension2_resources': dim2,
            'dimension3_explainability': dim3
        }
    
    config = PipelineConfig(
        test_mode=args.test_mode,
        sample_ratio=sample_ratio,
        random_state=args.random_state,
        output_dir=args.output_dir,
        interactive=args.interactive,
        ahp_preferences=ahp_preferences,
        cic_max_files=args.cic_max_files,
        synthetic_mode=args.synthetic
    )
    
    return config


if __name__ == "__main__":
    # Test CLI parsing
    import sys
    sys.argv = ['cli.py', '--phase', '1', '--test-mode']
    args = parse_args()
    config = args_to_config(args)
    print(f"✅ CLI parsing OK")
    print(f"  Phase: {args.phase}")
    print(f"  Test mode: {config.test_mode}")
    print(f"  Sample ratio: {config.sample_ratio}")
