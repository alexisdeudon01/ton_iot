import os
import pytest
import polars as pl
from src.core.dag.context import DAGContext
from src.app.pipeline.tasks.t01_consolidate_cic import T01_ConsolidateCIC
from src.infra.io.polars_io import PolarsIO
from src.infra.artifacts.file_store import FileArtifactStore
from src.infra.events.queue_bus import QueueEventBus
from src.infra.logging.structured_logger import StructuredEventLogger
from src.core.contracts.config import PipelineConfig

def test_t01_smoke(tmp_path):
    # Create dummy CSV
    csv_dir = tmp_path / "cic"
    csv_dir.mkdir()
    csv_file = csv_dir / "test.csv"
    df = pl.DataFrame({"Label": ["BENIGN", "DDoS"], "feat1": [1, 2]})
    df.write_csv(csv_file)
    
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    art_dir = work_dir / "artifacts"
    art_dir.mkdir()
    
    config = PipelineConfig(
        paths={
            "cic_dir_path": str(csv_dir),
            "ton_csv_path": "dummy",
            "work_dir": str(work_dir),
            "artifacts_dir": str(art_dir)
        },
        io={},
        sampling_policy={},
        alignment={},
        preprocessing={},
        clustering={},
        training={"algorithms": ["LR", "DT", "RF", "CNN", "TabNet"]},
        fusion={}
    )
    
    bus = QueueEventBus()
    context = DAGContext(
        run_id="test",
        config=config,
        artifact_store=FileArtifactStore(str(art_dir)),
        event_bus=bus,
        logger=StructuredEventLogger(bus, "test"),
        table_io=PolarsIO()
    )
    
    task = T01_ConsolidateCIC("T01")
    result = task.run(context)
    
    assert result.status == "ok"
    assert os.path.exists(os.path.join(work_dir, "data", "cic_consolidated.parquet"))
    bus.stop()
