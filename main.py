import os
import yaml
import polars as pl
import numpy as np
import tkinter as tk
from threading import Thread
from typing import List, Dict

# Import des modules du projet
from config.schema import PipelineConfig
from core.memory_manager import MemoryManager
from core.artifacts import ArtifactManager
from core.logger import pipeline_logger
from preprocessing.cleaner import DataCleaner
from preprocessing.preprocess_builder import PreprocessBuilder
from alignment.feature_alignment import FeatureAligner
from alignment.selectors import FeatureSelector
from models.lr import LRModel
from models.dt import DTModel
from models.rf import RFModel
from models.cnn import CNNModel
from models.tabnet import TabNetModel
from fusion.late_fusion import LateFusion
from xai.shap_values import ShapExplainer
from xai.metrics import XAIMetrics
from evaluation.performance import PerformanceEvaluator
from evaluation.resources import ResourceTracker
from gui.app import DDoSApp

def run_pipeline(config: PipelineConfig, app: DDoSApp):
    """
    Orchestrateur principal du pipeline DDoS.
    Exécuté dans un thread séparé pour maintenir la réactivité de la GUI.
    """
    try:
        tracker = ResourceTracker()
        tracker.start()
        
        mem_manager = MemoryManager(config.memory)
        artifact_manager = ArtifactManager(config.paths.output_dir)
        
        app.log("🚀 Démarrage du pipeline DDoS multi-datasets...")
        app.update_pipeline("Initialisation", 5)

        # 1. Chargement et Nettoyage
        app.log("📥 Chargement et nettoyage des données...")
        app.update_pipeline("Chargement des données", 10)
        
        # CIC-DDoS2019
        cic_files = [os.path.join(config.paths.cic_raw_dir, f) for f in os.listdir(config.paths.cic_raw_dir) if f.endswith('.csv')]
        if not cic_files:
            raise FileNotFoundError("Aucun fichier CSV trouvé pour CIC.")
            
        cic_lf = pl.scan_csv(cic_files[0], ignore_errors=True) # Simplification pour démo
        if config.test_mode: cic_lf = cic_lf.head(2000)
        cic_lf = DataCleaner.clean_cic(cic_lf, cic_files[0])
        cic_df = cic_lf.collect()
        
        # TON-IoT
        ton_lf = pl.scan_csv(config.paths.ton_raw_path)
        if config.test_mode: ton_lf = ton_lf.head(2000)
        ton_lf = DataCleaner.clean_ton(ton_lf, config.paths.ton_raw_path)
        ton_df = ton_lf.collect()

        app.update_data_stats({
            "CIC Rows": cic_df.height,
            "TON Rows": ton_df.height,
            "CIC Cols": cic_df.width,
            "TON Cols": ton_df.width
        })
        app.log(f"Données chargées. CIC: {cic_df.height} lignes, TON: {ton_df.height} lignes.")

        # 2. Alignement des Features
        app.log("🔗 Alignement statistique des caractéristiques...")
        app.update_pipeline("Alignement des Features", 25)
        aligner = FeatureAligner(config.alignment)
        common_features = aligner.compute_alignment_matrix(cic_df, ton_df)
        
        if not common_features:
            app.log("⚠️ Aucune feature commune alignée trouvée. Utilisation d'un fallback.")
            common_features = [c for c in cic_df.columns if c in ton_df.columns and c not in ["y", "source_file"]][:5]

        app.log(f"Features alignées ({len(common_features)}) : {common_features[:5]}...")
        
        cic_df = FeatureSelector.project_to_common(cic_df, common_features)
        ton_df = FeatureSelector.project_to_common(ton_df, common_features)

        # 3. Prétraitement
        app.log("🛠️ Prétraitement des données (Imputation, Scaling)...")
        app.update_pipeline("Prétraitement", 40)
        builder = PreprocessBuilder(config.preprocessing)
        cic_df, feature_order = builder.execute(cic_df)
        ton_df, _ = builder.execute(ton_df)
        
        artifact_manager.save_table(cic_df, "cic_preprocessed")
        artifact_manager.save_table(ton_df, "ton_preprocessed")

        # 4. Entraînement des Modèles (Exemple avec RF et LR)
        app.log("🧠 Entraînement des modèles ML...")
        app.update_pipeline("Entraînement", 60)
        
        X_cic = cic_df.select(feature_order).to_numpy()
        y_cic = cic_df["y"].to_numpy()
        
        # On entraîne un Random Forest pour la démo
        rf = RFModel(feature_order)
        rf.fit(X_cic, y_cic)
        artifact_manager.save_model(rf.model, "model_rf_cic")
        app.log("✅ Random Forest entraîné sur CIC.")

        # 5. Late Fusion (Simulation)
        app.log("🧪 Application de la Late Fusion...")
        app.update_pipeline("Late Fusion", 80)
        X_ton = ton_df.select(feature_order).to_numpy()
        y_ton = ton_df["y"].to_numpy()
        
        p_cic = rf.predict_proba(X_ton)
        # Simulation d'un second modèle pour la fusion
        p_ton = p_cic * 0.9 + 0.05 
        
        fused_proba = LateFusion.weighted_average([p_cic, p_ton], [0.6, 0.4])
        y_pred = LateFusion.get_decision(fused_proba, config.fusion.threshold)
        
        metrics = PerformanceEvaluator.compute_metrics(y_ton, y_pred)
        app.log(f"Performance Fusion - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        # 6. Explicabilité XAI
        app.log("🔍 Analyse d'explicabilité SHAP...")
        app.update_pipeline("Analyse XAI", 90)
        background = X_cic[:50]
        explainer = ShapExplainer(rf.model.predict_proba, background)
        shap_vals = explainer.explain(X_ton[:10])
        global_imp = explainer.compute_global_importance(shap_vals)
        
        app.update_xai({
            "features": feature_order[:10],
            "importances": global_imp[:10].tolist()
        })

        # 7. Finalisation
        usage = tracker.get_usage()
        app.update_resources(usage["cpu_percent"], usage["ram_mb"])
        app.log(f"✨ Pipeline terminé avec succès en {usage['duration_s']:.2f}s")
        app.update_pipeline("Terminé", 100)

    except Exception as e:
        app.log(f"❌ ERREUR CRITIQUE : {str(e)}")
        pipeline_logger.error("pipeline_failure", str(e))

def main():
    # Chargement de la configuration
    config_path = "config/pipeline.yaml"
    if not os.path.exists(config_path):
        print(f"Erreur: {config_path} non trouvé. Veuillez créer le fichier de configuration.")
        return

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = PipelineConfig(**config_dict)

    # Initialisation de l'interface graphique
    root = tk.Tk()
    app = DDoSApp(root)
    
    # Lancement du pipeline dans un thread worker
    pipeline_thread = Thread(target=run_pipeline, args=(config, app))
    pipeline_thread.daemon = True
    pipeline_thread.start()
    
    # Boucle principale Tkinter
    root.mainloop()

if __name__ == "__main__":
    main()
