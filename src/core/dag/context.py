from typing import Dict, Any
from src.core.contracts.config import PipelineConfig
from src.core.ports.artifacts import ArtifactPort
from src.core.ports.event_bus import EventBus
from src.core.ports.logging import StructuredLogger
from src.core.ports.table_io import TableIO

class DAGContext:
    def __init__(
        self,
        run_id: str,
        config: PipelineConfig,
        artifact_store: ArtifactPort,
        event_bus: EventBus,
        logger: StructuredLogger,
        table_io: TableIO
    ):
        self.run_id = run_id
        self.config = config
        self.artifact_store = artifact_store
        self.event_bus = event_bus
        self.logger = logger
        self.table_io = table_io
        self.shared_data: Dict[str, Any] = {}
