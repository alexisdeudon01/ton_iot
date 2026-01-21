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

from src.core.feature_categorization import categorize_features, get_category_scores
from src.core.dependency_graph import generate_er_dependency_diagram
from src.new_pipeline.config import ALGORITHMS, CIC_DDOS_DIR, RR_DIR, TON_IOT_PATH
from src.new_pipeline.data_loader import RealDataLoader
from src.new_pipeline.tester import PipelineTester
from src.new_pipeline.trainer import PipelineTrainer
from src.new_pipeline.validator import PipelineValidator
from src.new_pipeline.xai_manager import XAIManager
from src.system_monitor import SystemMonitor


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
    # 0. Initialize Dask Client for resource management
    # We limit memory per worker to stay within 50% total RAM
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    memory_limit = f"{total_ram_gb * 0.45:.1f}GB" # 45% to be safe

    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=2,
        memory_limit=memory_limit,
        dashboard_address=":8787"
    )
    client = Client(cluster)

    # Initialize System Monitor for plotting
    monitor = SystemMonitor(max_memory_percent=50.0)
    monitor.start_monitoring(interval=0.1)

    setup_logging()
    logger = logging.getLogger(__name__)

    print("\n" + "#" * 80)
    print("### PIPELINE DDOS SENIOR EXPERT V8 - DASK POWERED & RESOURCE MANAGED ###")
    print(f"### DASK CLIENT: {client.dashboard_link} ###")
    print(f"### RAM LIMIT: 50% ({memory_limit}) | DASK OUT-OF-CORE ###")
    print(f"### SAMPLE RATIO: {sample_ratio*100:.4f}% ###")
    print("#" * 80)

    try:
        # Clean rr directory
        if RR_DIR.exists():
            shutil.rmtree(RR_DIR)
        RR_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Phase 1: Data Loading
        monitor.set_phase("Phase 1: Data Loading")
        loader = RealDataLoader(monitor, rr_dir=RR_DIR)

        # Load ToN-IoT and CICDDoS2019
        loader.load_datasets(TON_IOT_PATH, CIC_DDOS_DIR, sample_ratio=sample_ratio)

        loader.profile_and_validate()
        train_ddf, val_ddf, test_ddf = loader.get_splits()

        # Feature Categorization (on a sample for column names)
        print("\n" + "-" * 40)
        print("MICRO-TÂCHE: Catégorisation des features")
        all_features = [
            c
            for c in train_ddf.columns
            if c not in ["is_ddos", "label", "type", "dataset"]
        ]
        categorized = categorize_features(all_features)
        cat_scores = get_category_scores(categorized)
        print(f"RÉSULTAT: Features catégorisées en {len(categorized)} groupes.")
        print(f"SCORES CATÉGORIES: {cat_scores}")

        # For training, we might need to compute if models don't support Dask
        X_train = train_ddf[all_features]
        y_train = train_ddf["is_ddos"]
        X_val = val_ddf[all_features]
        y_val = val_ddf["is_ddos"]
        X_test = test_ddf[all_features]
        y_test = test_ddf["is_ddos"]

        # 2. Iterative Model Evaluation Loop
        results = {}

        for algo in ALGORITHMS:
            print("\n" + "=" * 80)
            print(f"ÉVALUATION COMPLÈTE DE L'ALGORITHME: {algo}")
            print("=" * 80)

            # 2.1 Training
            monitor.set_phase(f"Training {algo}")
            trainer = PipelineTrainer(random_state=42)
            trainer.train_single(algo, X_train, y_train)

            # 2.2 Validation & Tuning
            monitor.set_phase(f"Validation {algo}")
            validator = PipelineValidator(trainer.models, random_state=42)
            validator.validate_tuning(X_val, y_val, RR_DIR, algo_name=algo)

            # 2.3 XAI
            monitor.set_phase(f"XAI {algo}")
            xai = XAIManager(rr_dir=RR_DIR)
            xai.validate_xai(trainer.models, X_test, y_test, algo_name=algo)

            # 2.4 Testing
            monitor.set_phase(f"Testing {algo}")
            tester = PipelineTester(trainer.models, rr_dir=RR_DIR)
            tester.evaluate_all(X_test, y_test, algo_name=algo)

            results[algo] = {
                "trainer": trainer,
                "validator": validator,
                "xai": xai,
                "tester": tester,
            }

        # 3. Final Visualizations & Resource Plots
        monitor.set_phase("Finalizing")
        print("\n" + "-" * 40)
        print("MICRO-TÂCHE: Génération des graphiques finaux")

        # ER Diagram
        generate_er_dependency_diagram(RR_DIR / "pipeline_er_diagram.png")

        # Resource consumption
        monitor.plot_resource_consumption(str(RR_DIR / "resource_consumption.png"))
        monitor.generate_timeline_heatmap(str(RR_DIR / "execution_timeline.png"))

        # 3.1 Correlation Matrix on a sample of data
        X_sample = X_test.head(1000)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            X_sample.corr(), cmap="coolwarm", annot=False
        )
        plt.title("Correlation Matrix (Sample)")
        plt.savefig(RR_DIR / "correlation_matrix.png")
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
            plt.savefig(RR_DIR / "feature_importance_heatmap.png")
            plt.close()

        # 3.3 Category Metrics Visualization
        cat_metrics_df = pd.DataFrame(cat_scores, index=["Score"]).T
        cat_metrics_df.plot(kind="bar", figsize=(10, 6))
        plt.title("Feature Category Metrics (Performance, Explainability, Resources)")
        plt.ylabel("Score (1-10)")
        plt.xticks(rotation=45)
        plt.grid(axis="y", alpha=0.3)
        plt.savefig(RR_DIR / "category_metrics.png")
        plt.close()

        print(f"RÉSULTAT: Graphiques sauvegardés dans {RR_DIR}")

        print("\n" + "#" * 80)
        print(f"### PIPELINE TERMINÉ AVEC SUCCÈS ###")
        print(f"### TOUS LES RÉSULTATS SONT DANS: {RR_DIR} ###")
        print("#" * 80)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
    finally:
        monitor.stop_monitoring()
        client.close()
        cluster.close()


if __name__ == "__main__":
    # Default to 100% data unless specified
    ratio = 1.0
    if "--test-mode" in sys.argv:
        ratio = 0.001
    main(sample_ratio=ratio)
