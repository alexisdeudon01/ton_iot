import logging
import sys
import os
import shutil
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

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("new_pipeline.log")
        ]
    )

def manage_rr_dir():
    """Crée le dossier rr à la racine et le vide s'il existe déjà."""
    if RR_DIR.exists():
        logger = logging.getLogger(__name__)
        logger.info(f"Nettoyage du dossier {RR_DIR}...")
        shutil.rmtree(RR_DIR)
    RR_DIR.mkdir(parents=True, exist_ok=True)

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("=== DÉMARRAGE DU PIPELINE DDOS SENIOR EXPERT ===")

    # Gestion du dossier rr
    manage_rr_dir()

    # 1. Phase 1: Data Loading, Profiling & Validation
    if not DATA_PATH.exists():
        logger.error(f"Dataset non trouvé à {DATA_PATH}. Création d'un dataset synthétique pour la démo.")
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        n_samples = 5000
        df_demo = pd.DataFrame(np.random.randn(n_samples, 20), columns=[f'feat_{i}' for i in range(20)])
        df_demo['label'] = np.random.randint(0, 2, n_samples)
        df_demo.to_csv(DATA_PATH, index=False)

    loader = RealDataLoader(DATA_PATH, rr_dir=RR_DIR)
    profiling_report = loader.load_and_profile()

    # Échantillonnage si trop volumineux pour SVM/TabNet dans une démo rapide
    if profiling_report['total_rows'] > 20000:
        logger.info("Dataset volumineux détecté (>20k). Échantillonnage à 20000 lignes pour la démo.")
        loader.df = loader.df.sample(20000, random_state=42)
        loader.load_and_profile() # Re-profiling

    train_df, val_df, test_df = loader.get_splits()

    # 2. Phase 2: Training (DT, RF, CNN, LR, TabNet)
    trainer = PipelineTrainer(random_state=42)
    y_train = train_df['is_ddos']
    X_train = train_df.drop(['is_ddos', 'label', 'type'], axis=1, errors='ignore')

    trainer.train_all(X_train, y_train)
    trainer.plot_results(RR_DIR)

    # 3. Phase 3: Validation (Tuning Dynamique)
    validator = PipelineValidator(trainer.models, random_state=42)
    y_val = val_df['is_ddos']
    X_val = val_df.drop(['is_ddos', 'label', 'type'], axis=1, errors='ignore')
    validator.validate_tuning(X_val, y_val, RR_DIR)

    # 4. Phase 4: XAI Manager (Validation & Sélection)
    xai = XAIManager(rr_dir=RR_DIR)
    y_test = test_df['is_ddos']
    X_test = test_df.drop(['is_ddos', 'label', 'type'], axis=1, errors='ignore')

    best_xai_methods = xai.validate_xai(trainer.models, X_test, y_test)
    xai.generate_visualizations(trainer.models, X_test)

    # 5. Phase 5: Testing (Évaluation Finale & Synthèse)
    tester = PipelineTester(trainer.models, rr_dir=RR_DIR)
    tester.evaluate_all(X_test, y_test)

    logger.info(f"=== PIPELINE TERMINÉ AVEC SUCCÈS. TOUS LES GRAPHIQUES SONT DANS {RR_DIR} ===")

if __name__ == "__main__":
    main()
