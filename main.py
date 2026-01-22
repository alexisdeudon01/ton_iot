#!/usr/bin/env python3
"""
Main entry point for Llama 4 Maverick DDoS Pipeline
Clean Architecture + Task DAG + Event Bus + Tkinter GUI
"""

import sys
import argparse
from pathlib import Path

# Configuration du chemin src
sys.path.insert(0, str(Path(__file__).parent))

def parse_args():
    parser = argparse.ArgumentParser(description="Llama 4 Maverick DDoS Pipeline")
    parser.add_argument("--gui", action="store_true", default=True, help="Launch Tkinter GUI (default)")
    parser.add_argument("--cli", action="store_false", dest="gui", help="Run in CLI mode")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml", help="Path to config file")
    parser.add_argument("--test-mode", action="store_true", help="Run with sampled data for testing")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.gui:
        print("🚀 Launching GUI...")
        # Note: In a real app, we'd pass test_mode to the GUI controller
        from src.app.ui.main import main as run_ui
        run_ui()
    else:
        print(f"⚙️ Running in CLI mode (test_mode={args.test_mode})...")
        from src.app.pipeline.main import run_pipeline
        run_pipeline(args.config, test_mode_override=args.test_mode)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
