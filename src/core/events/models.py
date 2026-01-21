from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Literal
from enum import Enum
import time

class EventType(str, Enum):
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    TASK_STARTED = "TASK_STARTED"
    TASK_FINISHED = "TASK_FINISHED"
    LOG_LINE = "LOG_LINE"
    TABLE_PROFILE_READY = "TABLE_PROFILE_READY"
    DISTRIBUTION_READY = "DISTRIBUTION_READY"
    RESOURCE_SNAPSHOT = "RESOURCE_SNAPSHOT"
    PIPELINE_GRAPH_READY = "PIPELINE_GRAPH_READY"
    ARTIFACT_CREATED = "ARTIFACT_CREATED"
    PROGRESS = "PROGRESS"
    STATUS = "STATUS"

class PipelineEvent(BaseModel):
    type: EventType
    payload: Dict[str, Any] = Field(default_factory=dict)
    ts: float = Field(default_factory=time.time)
    run_id: str
    task_name: Optional[str] = None
