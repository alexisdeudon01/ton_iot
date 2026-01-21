import logging
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import psutil
from dask.distributed import Client, LocalCluster

# Ensure project root is in sys.path
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.core.feature_categorization import (
    categorize_features,
    get_category_scores,
    print_verbose_feature_info,
)
from src.core.dependency_graph import generate_er_dependency_diagram
from src.datastructure.base import IRPDaskFrame, IRPDataFrame
from src.datastructure.flow import NetworkFlow
from src.new_pipeline.config import config
from src.new_pipeline.data_loader import RealDataLoader
from src.new_pipeline.tester import PipelineTester
from src.new_pipeline.trainer import PipelineTrainer
from src.new_pipeline.validator import PipelineValidator
from src.new_pipeline.xai_manager import XAIManager
from src.system_monitor import SystemMonitor
from src.core.memory_manager import MemoryAwareProcessor
from src.evaluation.visualization_service import VisualizationService


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("new_pipeline.log"),
        ],
    )


def main(sample_ratio: float = 1.0):
    # 0. Initialize Support Layer (Layer 3)
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    memory_limit = f"{total_ram_gb * 0.45:.1f}GB"

    cluster = None
    client = None

    monitor = SystemMonitor(max_memory_percent=config.max_memory_percent)
    monitor.start_monitoring(interval=0.1)
    
    memory_mgr = MemoryAwareProcessor(safety_margin=0.7)
    viz_service = VisualizationService(config.rr_dir)

    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        cluster = LocalCluster(
            n_workers=config.dask_workers,
            threads_per_worker=2,
            memory_limit=memory_limit,
            dashboard_address=":8787"
        )
        client = Client(cluster)

        print("\n" + "#" * 80)
        print("### PIPELINE DDOS 3-LAYER ARCHITECTURE - PARQUET & JOBLIB ###")
        print(f"### DASK CLIENT: {client.dashboard_link} ###")
        print(f"### RAM LIMIT: {config.max_memory_percent}% | DASK OUT-OF-CORE ###")
        print(f"### SAMPLE RATIO: {sample_ratio*100:.4f}% ###")
        print("#" * 80)

        # Clean rr directory
        if config.rr_dir.exists():
            shutil.rmtree(config.rr_dir)
        config.rr_dir.mkdir(parents=True, exist_ok=True)

        # 1. Phase 1: Data Loading (Layer 1 using Layer 3)
        monitor.set_phase("Phase 1: Data Loading")
        loader = RealDataLoader(monitor, rr_dir=config.rr_dir, parquet_dir=config.parquet_dir)

        # Load with Parquet optimization
        loader.load_datasets(config.ton_iot_path, config.cic_ddos_dir, sample_ratio=sample_ratio)

        loader.profile_and_validate()
        train_ddf, val_ddf, test_ddf = loader.get_splits()

        # Feature Categorization (on a sample for column names)
        all_features = [
            c
            for c in train_ddf.columns
            if c not in ["is_ddos", "label", "type", "dataset"]
        ]
        categorized = categorize_features(all_features)
        cat_scores = get_category_scores(categorized)

        # Verbose Expert Prompt
        print_verbose_feature_info(categorized, normalization_method="RobustScaler")

        # Data Structure Demonstration: NetworkFlow
        print("\n" + "-" * 40)
        print("MICRO-TÂCHE: Démonstration de la structure de données FLOW")
        sample_row = test_ddf.head(1).iloc[0]
        flow = NetworkFlow(
            flow_id="sample_flow_001",
            source_ip=str(sample_row.get("src_ip", "N/A")),
            dest_ip=str(sample_row.get("dst_ip", "N/A"))
        )
        flow.add_packet(sample_row)
        print(f"RÉSULTAT: {flow}")

        # For training, we might need to compute if models don't support Dask
        X_train = train_ddf[all_features]
        y_train = train_ddf["is_ddos"]
        X_val = val_ddf[all_features]
        y_val = val_ddf["is_ddos"]
        X_test = test_ddf[all_features]
        y_test = test_ddf["is_ddos"]

        # 2. Iterative Model Evaluation Loop
        results = {}

        for algo in config.algorithms:
            print("\n" + "=" * 80)
            print(f"ÉVALUATION COMPLÈTE DE L'ALGORITHME: {algo}")
            print("=" * 80)

            # 2.1 Training
            monitor.set_phase(f"Training {algo}")
            trainer = PipelineTrainer(memory_mgr, viz_service, config.models_dir, config.random_state)
            train_res = trainer.train_single(algo, X_train, y_train)

            # 2.2 Validation & Tuning
            monitor.set_phase(f"Validation {algo}")
            validator = PipelineValidator(trainer.models, memory_mgr, viz_service, config.random_state)
            val_res = validator.validate_tuning(X_val, y_val, algo_name=algo)

            # 2.3 XAI
            monitor.set_phase(f"XAI {algo}")
            xai = XAIManager(memory_mgr, viz_service, config.rr_dir)
            xai.validate_xai(trainer.models, X_test, y_test, algo_name=algo)

            # 2.4 Testing
            monitor.set_phase(f"Testing {algo}")
            tester = PipelineTester(trainer.models, memory_mgr, viz_service, config.rr_dir)
            test_res = tester.evaluate_all(X_test, y_test, algo_name=algo)

            results[algo] = {
                "trainer": trainer,
                "validator": validator,
                "xai": xai,
                "tester": tester,
                "train_res": train_res,
                "test_res": test_res
            }

        # 3. Final Visualizations & Resource Plots
        monitor.set_phase("Finalizing")
        print("\n" + "-" * 40)
        print("MICRO-TÂCHE: Génération des graphiques finaux")

        # ER Diagram
        generate_er_dependency_diagram(config.rr_dir / "pipeline_er_diagram.png")

        # Resource consumption
        monitor.plot_resource_consumption(str(config.rr_dir / "resource_consumption.png"))
        monitor.generate_timeline_heatmap(str(config.rr_dir / "execution_timeline.png"))

        # 3.1 Correlation Matrix on a sample of data
        X_sample = memory_mgr.safe_compute(X_test.head(1000), "correlation_matrix")
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            X_sample.corr(), cmap="coolwarm", annot=False
        )
        plt.title("Correlation Matrix (Sample)")
        plt.savefig(config.rr_dir / "correlation_matrix.png")
        plt.close()

        # 3.2 Feature Importance Heatmap (Across Algos)
        fi_data = {}
        for algo, res in results.items():
            if algo in ["RF", "DT"]:
                fi_data[algo] = res["trainer"].models[algo].feature_importances_

        if fi_data:
            fi_df = pd.DataFrame(fi_data, index=all_features)
            plt.figure(figsize=(12, 15))
            sns.heatmap(
                fi_df.sort_values(by="RF", ascending=False).head(30),
                annot=True,
                cmap="YlGnBu",
            )
            plt.title("Top 30 Feature Importances Comparison")
            plt.savefig(config.rr_dir / "feature_importance_heatmap.png")
            plt.close()

        # 3.3 Category Metrics Visualization
        cat_metrics_df = pd.DataFrame(cat_scores, index=["Score"]).T
        cat_metrics_df.plot(kind="bar", figsize=(10, 6))
        plt.title("Feature Category Metrics (Performance, Explainability, Resources)")
        plt.ylabel("Score (1-10)")
        plt.xticks(rotation=45)
        plt.grid(axis="y", alpha=0.3)
        plt.savefig(config.rr_dir / "category_metrics.png")
        plt.close()

        print(f"RÉSULTAT: Graphiques sauvegardés dans {config.rr_dir}")

        print("\n" + "#" * 80)
        print(f"### PIPELINE TERMINÉ AVEC SUCCÈS ###")
        print(f"### TOUS LES RÉSULTATS SONT DANS: {config.rr_dir} ###")
        print("#" * 80)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
    finally:
        monitor.stop_monitoring()
        if client: client.close()
        if cluster: cluster.close()


if __name__ == "__main__":
    # Default to 100% data unless specified
    ratio = 1.0
    if "--test-mode" in sys.argv:
        ratio = 0.001
    main(sample_ratio=ratio)
