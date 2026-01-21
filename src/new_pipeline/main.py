import logging
import sys
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import dask.dataframe as dd

from src.new_pipeline.config import config
from src.new_pipeline.data_loader import LateFusionDataLoader
from src.new_pipeline.preprocessor import DatasetPreprocessor
from src.new_pipeline.feature_selector import FeatureSelector
from src.new_pipeline.trainer import LateFusionTrainer
from src.new_pipeline.fusion_manager import LateFusionManager
from src.new_pipeline.evaluator import LateFusionEvaluator
from src.new_pipeline.xai_manager import LateFusionXAI

def check_dependencies():
    """
    Vérifie les dépendances critiques avant le lancement.
    CORRECTION: Extension de la vérification aux bibliothèques réellement utilisées.
    """
    import importlib
    deps = ["dask", "pandas", "numpy", "sklearn", "torch", "shap", "joblib", "pydantic", "pyarrow"]
    missing = []
    for dep in deps:
        try:
            importlib.import_module(dep)
        except ImportError:
            missing.append(dep)
    
    if missing:
        print(f"CRITICAL: Missing dependencies: {', '.join(missing)}")
        sys.exit(1)
    print("✓ All dependencies satisfied.")

def setup_logging():
    config.paths.logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.paths.run_log),
        ],
    )
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="TON IoT Late Fusion ML Pipeline")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode (0.1%% data)")
    parser.add_argument("--sample-ratio", type=float, default=1.0, help="Data sampling ratio (0.0 to 1.0)")
    parser.add_argument("--force-prepare", action="store_true", help="Force CSV to Parquet conversion")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Sync config with CLI args (CORRECTION)
    config.test_mode = args.test_mode
    sample_ratio = 0.001 if args.test_mode else args.sample_ratio
    
    check_dependencies()
    logger = setup_logging()
    
    logger.info("="*50)
    logger.info("STARTING COMPLETE LATE FUSION DDoS PIPELINE")
    logger.info(f"Mode: {'TEST (0.1%)' if config.test_mode else f'NORMAL ({sample_ratio*100}%)'}")
    logger.info("="*50)
    
    start_total = time.time()

    try:
        # Ensure output directories exist
        config.paths.out_root.mkdir(parents=True, exist_ok=True)
        config.paths.artifacts_root.mkdir(parents=True, exist_ok=True)
        config.paths.reports_root.mkdir(parents=True, exist_ok=True)

        # PHASE 0 & 1: Data Loading & Alignment
        loader = LateFusionDataLoader()
        # CORRECTION: On passe le sample_ratio au loader pour Dask
        loader.prepare_datasets(force=args.force_prepare)
        f_common = loader.align_features()

        # PHASE 2: Preprocessing (Par Dataset)
        prep_cic = DatasetPreprocessor("cic")
        prep_ton = DatasetPreprocessor("ton")
        
        # CORRECTION: Chargement Dask optimisé selon le mode
        ddf_cic = dd.read_parquet(config.paths.cic_parquet)
        ddf_ton = dd.read_parquet(config.paths.ton_parquet)
        
        if sample_ratio < 1.0:
            logger.info(f"Sampling datasets at {sample_ratio*100}%...")
            ddf_cic = ddf_cic.sample(frac=sample_ratio, random_state=42)
            ddf_ton = ddf_ton.sample(frac=sample_ratio, random_state=42)
        
        paths_cic = prep_cic.process(ddf_cic, f_common)
        paths_ton = prep_ton.process(ddf_ton, f_common)

        # PHASE 3: Feature Selection
        selector = FeatureSelector()
        f_fusion = selector.select_features(paths_cic['train'], paths_ton['train'], f_common)

        # PHASE 4: Training Séparé
        trainer_cic = LateFusionTrainer("cic")
        trainer_ton = LateFusionTrainer("ton")
        
        res_cic = trainer_cic.train(paths_cic['train'], paths_cic['val'], config.phase4.model_cic, f_fusion)
        res_ton = trainer_ton.train(paths_ton['train'], paths_ton['val'], config.phase4.model_ton, f_fusion)

        if not res_cic.success or not res_ton.success:
            raise RuntimeError("Training failed for one or more models.")

        # PHASE 5: Late Fusion Optimization
        fusion_mgr = LateFusionManager()
        best_w = fusion_mgr.optimize_fusion(
            paths_cic['val'], paths_ton['val'],
            res_cic.model_path, res_ton.model_path,
            f_fusion
        )

        # PHASE 6: Évaluation
        evaluator = LateFusionEvaluator(res_cic.model_path, res_ton.model_path, best_w, f_fusion)
        final_results = evaluator.evaluate_all(paths_cic['test'], paths_ton['test'])

        # PHASE 7: Ressources & XAI
        xai = LateFusionXAI(config.phase8.xai_artifacts_dir)
        m_cic = joblib.load(res_cic.model_path)
        df_test_cic = pd.read_parquet(paths_cic['test'])
        
        # Audit
        audit = xai.run_resource_audit(m_cic, df_test_cic[f_fusion].head(1000))
        logger.info(f"Resource Audit (CIC Model): {audit}")
        
        # XAI
        xai.run_xai(m_cic, df_test_cic[f_fusion], df_test_cic[config.phase0.label_col], "CIC_Model")

        # PHASE 8: Rapport Final
        logger.info("Starting Phase 8: Final Reporting")
        report_rows = []
        for res in final_results:
            report_rows.append({
                'model_name': res.model_name,
                'dataset': 'Combined' if 'FUSION' in res.model_name else res.model_name.split(' ')[0],
                'detection_performance': f"F1: {res.f1_score:.4f}, Recall: {res.recall:.4f}",
                'resource_efficiency': f"Latency: {audit['inference_latency_ms']:.4f}ms" if 'CIC' in res.model_name else "N/A",
                'explainability': "XAI artifacts generated" if 'CIC' in res.model_name else "N/A",
                'label_balance': "Stratified",
                'notes': f"Fusion weight w={best_w:.2f}" if 'FUSION' in res.model_name else ""
            })
        
        report_df = pd.DataFrame(report_rows)
        report_df.to_csv(config.report.report_csv, index=False)
        logger.info(f"Final report exported to {config.report.report_csv}")

        logger.info("="*50)
        logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {time.time() - start_total:.2f}s")
        logger.info("="*50)

    except Exception as e:
        logger.error(f"CRITICAL: Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
