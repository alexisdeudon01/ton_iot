import logging
import sys
import os
import shutil
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.new_pipeline.config import RR_DIR, DATA_PATH
from src.new_pipeline.data_loader import RealDataLoader
from src.new_pipeline.trainer import PipelineTrainer
from src.new_pipeline.validator import PipelineValidator
from src.new_pipeline.xai_manager import XAIManager
from src.new_pipeline.tester import PipelineTester
from src.system_monitor import SystemMonitor

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("new_pipeline.log")
        ]
    )

def main():
    # 0. Initialize System Monitor with 50% RAM limit
    monitor = SystemMonitor(max_memory_percent=50.0)
    monitor.start_monitoring(interval=0.2)

    setup_logging()
    logger = logging.getLogger(__name__)

    print("\n" + "#"*80)
    print("### PIPELINE DDOS SENIOR EXPERT V5 - RESOURCE AWARE ###")
    print("### RAM LIMIT: 50% | MULTI-THREADED LOADING ###")
    print("#"*80)

    try:
        # Clean rr directory
        if RR_DIR.exists():
            shutil.rmtree(RR_DIR)
        RR_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Phase 1: Data Loading (100% Data)
        monitor.set_phase("Data Loading")
        loader = RealDataLoader(DATA_PATH.parent, monitor, rr_dir=RR_DIR)
        # Load all CSVs in the directory using multi-threading
        loader.load_all_csv_multithreaded(sample_ratio=1.0)
        loader.profile_and_validate()
        train_df, val_df, test_df = loader.get_splits()

        X_train = train_df.drop(['is_ddos', 'label', 'type'], axis=1, errors='ignore')
        y_train = train_df['is_ddos']
        X_val = val_df.drop(['is_ddos', 'label', 'type'], axis=1, errors='ignore')
        y_val = val_df['is_ddos']
        X_test = test_df.drop(['is_ddos', 'label', 'type'], axis=1, errors='ignore')
        y_test = test_df['is_ddos']

        # 2. Phase 2: Training
        monitor.set_phase("Training")
        trainer = PipelineTrainer(random_state=42)
        trainer.train_all(X_train, y_train)
        trainer.plot_results(RR_DIR)

        # 3. Phase 3: Validation
        monitor.set_phase("Validation")
        validator = PipelineValidator(trainer.models, random_state=42)
        validator.validate_tuning(X_val, y_val, RR_DIR)

        # 4. Phase 4: XAI
        monitor.set_phase("XAI")
        xai = XAIManager(rr_dir=RR_DIR)
        xai.validate_xai(trainer.models, X_test, y_test)
        xai.generate_visualizations(trainer.models, X_test)

        # 5. Phase 5: Testing
        monitor.set_phase("Testing")
        tester = PipelineTester(trainer.models, rr_dir=RR_DIR)
        tester.evaluate_all(X_test, y_test)

        # 6. Resource Plots & Timeline
        monitor.set_phase("Finalizing")
        monitor.plot_resource_consumption(RR_DIR / "resource_consumption.png")
        monitor.generate_timeline_heatmap(RR_DIR / "execution_timeline.png")

        print("\n" + "#"*80)
        print(f"### PIPELINE TERMINÉ AVEC SUCCÈS ###")
        print(f"### TOUS LES RÉSULTATS SONT DANS: {RR_DIR} ###")
        print("#"*80)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
