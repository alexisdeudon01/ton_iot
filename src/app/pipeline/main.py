import uuid
import yaml
import os
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

from src.app.pipeline.registry import TaskRegistry

def build_pipeline_graph() -> DAGGraph:
    graph = DAGGraph()
    
    t00 = TaskRegistry.get_task_cls("T00_InitRun")("T00_InitRun")
    t01 = TaskRegistry.get_task_cls("T01_ConsolidateCIC")("T01_ConsolidateCIC")
    t02 = TaskRegistry.get_task_cls("T02_CleanTON")("T02_CleanTON")
    t03 = TaskRegistry.get_task_cls("T03_ProfileCIC")("T03_ProfileCIC", inputs=["cic_consolidated"])
    t04 = TaskRegistry.get_task_cls("T04_ProfileTON")("T04_ProfileTON", inputs=["ton_clean"])
    t05 = TaskRegistry.get_task_cls("T05_AlignFeatures")("T05_AlignFeatures", inputs=["cic_consolidated", "ton_clean"])
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

    graph.add_task(t00)
    graph.add_task(t01, depends_on=["T00_InitRun"])
    graph.add_task(t02, depends_on=["T00_InitRun"])
    graph.add_task(t03, depends_on=["T01_ConsolidateCIC"])
    graph.add_task(t04, depends_on=["T02_CleanTON"])
    graph.add_task(t05, depends_on=["T01_ConsolidateCIC", "T02_CleanTON"])
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
    
    return graph

def run_pipeline(config_path: str, event_bus: QueueEventBus = None, test_mode_override: bool = None):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    config = PipelineConfig.model_validate(config_dict)
    if test_mode_override is not None:
        config.test_mode = test_mode_override
        if config.test_mode:
            config.sample_ratio = 0.01 # Force small sample in test mode

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
