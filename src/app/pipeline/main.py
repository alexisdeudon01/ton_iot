import uuid
import yaml
import os
import sys
import shutil
from datetime import datetime

# Ajout du répertoire racine au PYTHONPATH pour éviter l'erreur ModuleNotFoundError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.core.contracts.config import PipelineConfig
from src.core.dag.graph import DAGGraph
from src.core.dag.runner import DAGRunner
from src.core.dag.context import DAGContext
from src.infra.io.polars_io import PolarsIO
from src.infra.artifacts.file_store import FileArtifactStore
from src.infra.events.queue_bus import QueueEventBus
from src.infra.logging.structured_logger import StructuredEventLogger

# Import tasks to register them
import src.app.pipeline.tasks.t00_init_run
import src.app.pipeline.tasks.t01_consolidate_cic
import src.app.pipeline.tasks.t02_clean_ton
import src.app.pipeline.tasks.t03_profile_cic
import src.app.pipeline.tasks.t04_profile_ton
import src.app.pipeline.tasks.t05_align_features
import src.app.pipeline.tasks.t05_feature_distribution
import src.app.pipeline.tasks.t06_project_cic
import src.app.pipeline.tasks.t07_project_ton
import src.app.pipeline.tasks.t08_build_preprocess_cic
import src.app.pipeline.tasks.t09_build_preprocess_ton
import src.app.pipeline.tasks.t12_train_cic
import src.app.pipeline.tasks.t13_train_ton
import src.app.pipeline.tasks.t14_predict_cic
import src.app.pipeline.tasks.t15_predict_ton
import src.app.pipeline.tasks.t16_late_fusion
import src.app.pipeline.tasks.t17_evaluate
import src.app.pipeline.tasks.t18_mcdm_decision

from src.app.pipeline.registry import TaskRegistry

def build_pipeline_graph() -> DAGGraph:
    print("\n" + "="*80)
    print("CONSTRUCTION DU GRAPH D'EXÉCUTION DU PIPELINE")
    print("="*80)
    
    tasks_info = [
        ("T00_InitRun", "Initialisation de l'environnement et des dossiers de travail"),
        ("T01_ConsolidateCIC", "Lecture et consolidation du dataset CIC-DDoS2019 (CSV -> Parquet)"),
        ("T02_CleanTON", "Nettoyage et formatage du dataset ToN-IoT"),
        ("T03_ProfileCIC", "Analyse statistique et profilage des données CIC"),
        ("T04_ProfileTON", "Analyse statistique et profilage des données ToN-IoT"),
        ("T05_AlignFeatures", "SÉLECTION DES FEATURES : Calcul de l'intersection des colonnes communes"),
        ("T05_FeatureDistribution", "VISUALISATION : Distributions post-preprocessing (15 features universelles)"),
        ("T06_ProjectCIC", "Projection du dataset CIC sur l'espace de caractéristiques commun"),
        ("T07_ProjectTON", "Projection du dataset ToN-IoT sur l'espace de caractéristiques commun"),
        ("T08_BuildPreprocessCIC", "PREPROCESSING : Construction du RobustScaler pour CIC (.joblib)"),
        ("T09_BuildPreprocessTON", "PREPROCESSING : Construction du RobustScaler pour ToN-IoT (.joblib)"),
        ("T12_TrainCIC", "ENTRAÎNEMENT : Apprentissage des 5 modèles sur les données CIC"),
        ("T13_TrainTON", "ENTRAÎNEMENT : Apprentissage des 5 modèles sur les données ToN-IoT"),
        ("T14_PredictCIC", "INFÉRENCE : Génération des prédictions sur le jeu de test CIC"),
        ("T15_PredictTON", "INFÉRENCE : Génération des prédictions sur le jeu de test ToN-IoT"),
        ("T16_LateFusion", "FUSION : Combinaison des résultats via Late Fusion (Moyenne pondérée)"),
        ("T17_Evaluate", "ÉVALUATION : Calcul des métriques de performance (F1, Précision, Rappel)"),
        ("T18_MCDM_Decision", "DÉCISION : Classement final via MOO-MCDM-MCDA (TOPSIS/Pareto)")
    ]

    for code, desc in tasks_info:
        print(f"  [+] {code.ljust(25)} : {desc}")

    graph = DAGGraph()
    
    t00 = TaskRegistry.get_task_cls("T00_InitRun")("T00_InitRun")
    t01 = TaskRegistry.get_task_cls("T01_ConsolidateCIC")("T01_ConsolidateCIC")
    t02 = TaskRegistry.get_task_cls("T02_CleanTON")("T02_CleanTON")
    t03 = TaskRegistry.get_task_cls("T03_ProfileCIC")("T03_ProfileCIC", inputs=["cic_consolidated"])
    t04 = TaskRegistry.get_task_cls("T04_ProfileTON")("T04_ProfileTON", inputs=["ton_clean"])
    t05 = TaskRegistry.get_task_cls("T05_AlignFeatures")("T05_AlignFeatures", inputs=["cic_consolidated", "ton_clean"])
    t05_dist = TaskRegistry.get_task_cls("T05_FeatureDistribution")(
        "T05_FeatureDistribution",
        inputs=["cic_projected", "ton_projected", "preprocess_cic", "preprocess_ton", "alignment_spec"],
    )
    t06 = TaskRegistry.get_task_cls("T06_ProjectCIC")("T06_ProjectCIC", inputs=["cic_consolidated", "alignment_spec"])
    t07 = TaskRegistry.get_task_cls("T07_ProjectTON")("T07_ProjectTON", inputs=["ton_clean", "alignment_spec"])
    t08 = TaskRegistry.get_task_cls("T08_BuildPreprocessCIC")("T08_BuildPreprocessCIC", inputs=["cic_projected"])
    t09 = TaskRegistry.get_task_cls("T09_BuildPreprocessTON")("T09_BuildPreprocessTON", inputs=["ton_projected"])
    t12 = TaskRegistry.get_task_cls("T12_TrainCIC")("T12_TrainCIC", inputs=["cic_projected", "preprocess_cic"])
    t13 = TaskRegistry.get_task_cls("T13_TrainTON")("T13_TrainTON", inputs=["ton_projected", "preprocess_ton"])
    t14 = TaskRegistry.get_task_cls("T14_PredictCIC")("T14_PredictCIC", inputs=["cic_projected", "model_cic_RF", "preprocess_cic"])
    t15 = TaskRegistry.get_task_cls("T15_PredictTON")("T15_PredictTON", inputs=["ton_projected", "model_ton_RF", "preprocess_ton"])
    t16 = TaskRegistry.get_task_cls("T16_LateFusion")("T16_LateFusion", inputs=["predictions_cic", "predictions_ton"])
    t17 = TaskRegistry.get_task_cls("T17_Evaluate")("T17_Evaluate", inputs=["predictions_fused"])
    t18 = TaskRegistry.get_task_cls("T18_MCDM_Decision")("T18_MCDM_Decision", inputs=["run_report"])

    graph.add_task(t00)
    graph.add_task(t01, depends_on=["T00_InitRun"])
    graph.add_task(t02, depends_on=["T00_InitRun"])
    graph.add_task(t03, depends_on=["T01_ConsolidateCIC"])
    graph.add_task(t04, depends_on=["T02_CleanTON"])
    graph.add_task(t05, depends_on=["T01_ConsolidateCIC", "T02_CleanTON"])
    graph.add_task(t05_dist, depends_on=["T08_BuildPreprocessCIC", "T09_BuildPreprocessTON"])
    graph.add_task(t06, depends_on=["T05_AlignFeatures"])
    graph.add_task(t07, depends_on=["T05_AlignFeatures"])
    graph.add_task(t08, depends_on=["T06_ProjectCIC"])
    graph.add_task(t09, depends_on=["T07_ProjectTON"])
    graph.add_task(t12, depends_on=["T08_BuildPreprocessCIC"])
    graph.add_task(t13, depends_on=["T09_BuildPreprocessTON"])
    graph.add_task(t14, depends_on=["T12_TrainCIC"])
    graph.add_task(t15, depends_on=["T13_TrainTON"])
    graph.add_task(t16, depends_on=["T14_PredictCIC", "T15_PredictTON"])
    graph.add_task(t17, depends_on=["T16_LateFusion"])
    graph.add_task(t18, depends_on=["T17_Evaluate"])
    
    return graph

def setup_output_directories():
    """
    Crée les répertoires racine pour graphs, reports et configurations d'algorithmes.
    Demande à l'utilisateur s'il souhaite archiver les anciens résultats.
    """
    base_dir = "output"
    sub_dirs = ["output", "log"]
    root_dirs = ["graph", "algorithm_configurations", "reports", "other"]
    
    print("\n" + "="*80)
    print("GESTION DES RÉPERTOIRES DE SORTIE")
    print("="*80)

    for sd in sub_dirs:
        path = os.path.join(base_dir, sd)
        old_path = os.path.join(path, "_old")
        os.makedirs(old_path, exist_ok=True)

    for rd in root_dirs:
        os.makedirs(rd, exist_ok=True)

    # Interaction utilisateur
    try:
        ans_archive = input("?> Souhaitez-vous archiver les anciens résultats et logs dans _old ? (o/n) : ").lower()
    except EOFError:
        ans_archive = 'n'

    if ans_archive == 'o':
        for sd in sub_dirs:
            path = os.path.join(base_dir, sd)
            old_path = os.path.join(path, "_old")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_sub = os.path.join(old_path, timestamp)
            
            if os.path.exists(path):
                files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
                if files:
                    os.makedirs(archive_sub, exist_ok=True)
                    for f in files:
                        shutil.move(os.path.join(path, f), os.path.join(archive_sub, f))
                    print(f"  [OK] Fichiers de {path} déplacés vers {archive_sub}")

    try:
        ans_clean = input("?> Souhaitez-vous effacer les graphiques existants ? (o/n) : ").lower()
    except EOFError:
        ans_clean = 'n'

    if ans_clean == 'o':
        # On cible les dossiers de graphiques connus (nouvelle arborescence)
        graph_dirs = ["graph"]
        for gd in graph_dirs:
            if os.path.exists(gd):
                shutil.rmtree(gd)
            os.makedirs(gd, exist_ok=True)
            print(f"  [OK] Graphiques dans {gd} effacés.")

    # Mettre a jour un index des repertoires dans reports/
    report_index = os.path.join("reports", "directories.md")
    with open(report_index, "w") as f:
        f.write("# Repertoires de sortie\n\n")
        f.write("- graph/ : graphiques (distributions + MCDM)\n")
        f.write("- graph/decision/variations/ : variations par seuil (ressource, performance, explicabilite)\n")
        f.write("- graph/algorithms/dtreeviz/ : visualisations arbres (dtreeviz)\n")
        f.write("- algorithm_configurations/ : JSON des algorithmes\n")
        f.write("- reports/ : rapports (run_report, final_report)\n")
        f.write("- reports/variations/ : rapports DOCX par seuil\n")
        f.write("- other/ : reserve pour elements oublies\n")

def run_pipeline(config_path: str, event_bus: QueueEventBus = None, test_mode_override: bool = None):
    # Initialisation des dossiers
    setup_output_directories()

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    config = PipelineConfig.model_validate(config_dict)
    if test_mode_override is not None:
        config.test_mode = test_mode_override

    run_id = str(uuid.uuid4())
    
    if event_bus is None:
        event_bus = QueueEventBus()
        
    logger = StructuredEventLogger(event_bus, run_id)
    artifact_store = FileArtifactStore(config.paths.artifacts_dir)
    table_io = PolarsIO()
    
    context = DAGContext(
        run_id=run_id,
        config=config,
        artifact_store=artifact_store,
        event_bus=event_bus,
        logger=logger,
        table_io=table_io
    )
    
    graph = build_pipeline_graph()
    runner = DAGRunner(graph, context)
    
    # Add a console subscriber for CLI mode
    def console_logger(event):
        if event.type == "LOG_LINE":
            print(f"[{event.payload.get('level', 'INFO')}] {event.payload.get('action', '')}: {event.payload.get('message', '')}")
        elif event.type == "TASK_STARTED":
            print(f"\n▶️ Starting Task: {event.task_name}")
        elif event.type == "TASK_FINISHED":
            status = event.payload.get('status')
            icon = "✅" if status in ["ok", "degraded"] else "❌"
            error = event.payload.get('error')
            error_msg = f" - Error: {error}" if error else ""
            print(f"{icon} Task {event.task_name} finished in {event.payload.get('duration_s', 0):.2f}s{error_msg}")

    event_bus.subscribe(console_logger)
    
    results = runner.run()
    
    # Stop event bus if we created it here (CLI mode)
    if isinstance(event_bus, QueueEventBus):
        # Give a small time for last events to be processed
        import time
        time.sleep(0.5)
        event_bus.stop()
        
    return results

if __name__ == "__main__":
    run_pipeline("configs/pipeline.yaml")
