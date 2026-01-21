import logging
import sys
import time
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

def main():
    logger = setup_logging()
    logger.info("="*50)
    logger.info("STARTING COMPLETE LATE FUSION DDoS PIPELINE")
    logger.info("="*50)
    start_total = time.time()

    try:
        # PHASE 0 & 1: Data Loading & Alignment
        loader = LateFusionDataLoader()
        loader.prepare_datasets()
        f_common = loader.align_features()

        # PHASE 2: Preprocessing (Par Dataset)
        prep_cic = DatasetPreprocessor("cic")
        prep_ton = DatasetPreprocessor("ton")
        
        ddf_cic = dd.read_parquet(config.paths.cic_parquet)
        ddf_ton = dd.read_parquet(config.paths.ton_parquet)
        
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

        # PHASE 8: Rapport Final (CSV Export)
        logger.info("Starting Phase 8: Final Reporting")
        report_rows = []
        for res in final_results:
            # Mocking resource/xai notes for the report
            report_rows.append({
                'model_name': res.model_name,
                'dataset': 'Combined' if 'FUSION' in res.model_name else res.model_name.split(' ')[0],
                'detection_performance': f"F1: {res.f1_score:.4f}, Recall: {res.recall:.4f}",
                'resource_efficiency': f"Latency: {audit['inference_latency_ms']:.4f}ms" if 'CIC' in res.model_name else "N/A",
                'explainability': "Permutation Importance generated" if 'CIC' in res.model_name else "N/A",
                'label_balance': "Balanced via sampling",
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
