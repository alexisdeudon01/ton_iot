import logging
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.new_pipeline.data_loader import RealDataLoader
from src.new_pipeline.trainer import PipelineTrainer
from src.new_pipeline.validator import PipelineValidator
from src.new_pipeline.xai_manager import XAIManager
from src.new_pipeline.tester import PipelineTester

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
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("=== DÉMARRAGE DU PIPELINE DDOS SENIOR EXPERT ===")

    output_dir = Path("output/new_pipeline_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Phase 1: Data Loading & Profiling
    # On utilise le dataset TON_IoT par défaut s'il existe
    data_path = "datasets/ton_iot/train_test_network.csv"
    if not Path(data_path).exists():
        logger.error(f"Dataset non trouvé à {data_path}. Création d'un dataset synthétique pour la démo.")
        data_path = "datasets/synthetic_demo.csv"
        # Création d'un dataset réaliste pour la démo
        n_samples = 2000
        df_demo = pd.DataFrame(np.random.randn(n_samples, 15), columns=[f'feature_{i}' for i in range(15)])
        df_demo['label'] = np.random.randint(0, 2, n_samples)
        df_demo.to_csv(data_path, index=False)

    loader = RealDataLoader(data_path)
    profiling_report = loader.load_and_profile()
    train_df, val_df, test_df = loader.get_splits()

    # 2. Phase 2: Training
    trainer = PipelineTrainer()
    y_train = train_df['is_ddos']
    X_train = train_df.drop(['is_ddos', 'label', 'type'], axis=1, errors='ignore')

    trainer.train_all(X_train, y_train)
    trainer.plot_results(output_dir)

    # 3. Phase 3: Validation (Hyperparameter Tuning)
    validator = PipelineValidator(trainer.models)
    y_val = val_df['is_ddos']
    X_val = val_df.drop(['is_ddos', 'label', 'type'], axis=1, errors='ignore')
    validator.validate_tuning(X_val, y_val, output_dir)

    # 4. Phase 4: XAI Manager
    # Configuration XAI : SHAP, LIME, Feature Importance (FI)
    xai = XAIManager(config_xai_methods=['SHAP', 'LIME', 'FI'])
    y_test = test_df['is_ddos']
    X_test = test_df.drop(['is_ddos', 'label', 'type'], axis=1, errors='ignore')

    # Validation automatique selon Fidélité, Stabilité, Vitesse
    best_xai_methods = xai.validate_xai(trainer.models, X_test, y_test, output_dir)

    # Génération des visualisations spécifiques (SHAP Summary, LIME local)
    xai.generate_final_plots(trainer.models, X_test, output_dir, best_xai_methods)

    # 5. Phase 5: Testing
    tester = PipelineTester(trainer.models)
    tester.evaluate_all(X_test, y_test, output_dir)

    logger.info(f"=== PIPELINE TERMINÉ AVEC SUCCÈS. RÉSULTATS DANS {output_dir} ===")

if __name__ == "__main__":
    main()
