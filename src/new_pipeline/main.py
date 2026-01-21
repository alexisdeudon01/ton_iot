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
    # 0. Initialize System Monitor with 50% RAM limit and background thread
    monitor = SystemMonitor(max_memory_percent=50.0)
    monitor.start_monitoring(interval=0.1) # High frequency for better plots

    setup_logging()
    logger = logging.getLogger(__name__)

    print("\n" + "#"*80)
    print("### PIPELINE DDOS SENIOR EXPERT V6 - FULL DATA & RESOURCE MANAGED ###")
    print("### RAM LIMIT: 50% | DEDICATED MONITORING THREAD | PROACTIVE GC ###")
    print("#"*80)

    try:
        # Clean rr directory
        if RR_DIR.exists():
            shutil.rmtree(RR_DIR)
        RR_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Phase 1: Data Loading (100% Data)
        monitor.set_phase("Phase 1: Data Loading")
        loader = RealDataLoader(DATA_PATH.parent, monitor, rr_dir=RR_DIR)

        # Load all CSVs using multi-threading (one thread per file)
        # We use 100% of data as requested
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
        monitor.set_phase("Phase 2: Training")
        trainer = PipelineTrainer(random_state=42)
        trainer.train_all(X_train, y_train)
        trainer.plot_results(RR_DIR)

        # 3. Phase 3: Validation
        monitor.set_phase("Phase 3: Validation")
        validator = PipelineValidator(trainer.models, random_state=42)
        validator.validate_tuning(X_val, y_val, RR_DIR)

        # 4. Phase 4: XAI
        monitor.set_phase("Phase 4: XAI")
        xai = XAIManager(rr_dir=RR_DIR)
        xai.validate_xai(trainer.models, X_test, y_test)
        xai.generate_visualizations(trainer.models, X_test)

        # 5. Phase 5: Testing
        monitor.set_phase("Phase 5: Testing")
        tester = PipelineTester(trainer.models, rr_dir=RR_DIR)
        tester.evaluate_all(X_test, y_test)

        # 6. Resource Plots & Timeline
        monitor.set_phase("Finalizing")
        print("\n" + "-"*40)
        print("MICRO-TÂCHE: Génération des graphiques de consommation des ressources")
        monitor.plot_resource_consumption(str(RR_DIR / "resource_consumption.png"))
        monitor.generate_timeline_heatmap(str(RR_DIR / "execution_timeline.png"))
        print(f"RÉSULTAT: Graphiques sauvegardés dans {RR_DIR}")

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
