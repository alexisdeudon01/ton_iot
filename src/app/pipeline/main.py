import yaml
import uuid
import threading
import argparse
from src.core.contracts.config import PipelineConfig
from src.core.dag.engine import TaskContext, DAGRunner
from src.infra.io.polars_adapter import PolarsTableIOAdapter
from src.infra.profiling.profiler_adapter import ProfilerAdapter
from src.infra.artifacts.file_store import FileArtifactStoreAdapter
from src.infra.events.queue_adapter import QueueEventBusAdapter
from src.app.pipeline.tasks import (
    T00_InitRun, T01_ConsolidateCIC, T02_CleanTON, T03_ProfileCIC, T04_ProfileTON,
    T05_AlignFeatures, T06_ProjectCICToCommon, T07_ProjectTONToCommon,
    T08_BuildPreprocessCIC, T12_TuneTrainCIC, T16_LateFusion, T17_Evaluate
)

class SimpleLogger:
    def log(self, level: str, message: str, **kwargs):
        print(f"[{level}] {message}")

def run_pipeline(config_path: str, event_bus: QueueEventBusAdapter):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    config = PipelineConfig(**config_dict)
    run_id = str(uuid.uuid4())[:8]
    
    ctx = TaskContext(
        config=config,
        io=PolarsTableIOAdapter(compression=config.parquet_compression, row_group_size=config.row_group_size),
        profiler=ProfilerAdapter(),
        artifacts=FileArtifactStoreAdapter(base_dir=config.artifacts_dir),
        event_bus=event_bus,
        logger=SimpleLogger(),
        run_id=run_id
    )
    
    runner = DAGRunner(ctx)
    runner.add_task(T00_InitRun("T00_InitRun"))
    runner.add_task(T01_ConsolidateCIC("T01_ConsolidateCIC", requires=["T00_InitRun"]))
    runner.add_task(T02_CleanTON("T02_CleanTON", requires=["T00_InitRun"]))
    runner.add_task(T03_ProfileCIC("T03_ProfileCIC", requires=["T01_ConsolidateCIC"]))
    runner.add_task(T04_ProfileTON("T04_ProfileTON", requires=["T02_CleanTON"]))
    runner.add_task(T05_AlignFeatures("T05_AlignFeatures", requires=["T03_ProfileCIC", "T04_ProfileTON"]))
    runner.add_task(T06_ProjectCICToCommon("T06_ProjectCICToCommon", requires=["T05_AlignFeatures"]))
    runner.add_task(T07_ProjectTONToCommon("T07_ProjectTONToCommon", requires=["T05_AlignFeatures"]))
    runner.add_task(T08_BuildPreprocessCIC("T08_BuildPreprocessCIC", requires=["T06_ProjectCICToCommon"]))
    runner.add_task(T12_TuneTrainCIC("T12_TuneTrainCIC", requires=["T08_BuildPreprocessCIC"]))
    runner.add_task(T16_LateFusion("T16_LateFusion", requires=["T12_TuneTrainCIC"]))
    runner.add_task(T17_Evaluate("T17_Evaluate", requires=["T16_LateFusion"]))
    
    runner.run_all()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--ui", action="store_true")
    args = parser.parse_args()
    
    event_bus = QueueEventBusAdapter()
    
    if args.ui:
        from src.app.ui.main import start_ui
        # Run pipeline in background thread
        thread = threading.Thread(target=run_pipeline, args=(args.config, event_bus), daemon=True)
        thread.start()
        # Start UI in main thread
        start_ui(event_bus.get_queue())
    else:
        run_pipeline(args.config, event_bus)
