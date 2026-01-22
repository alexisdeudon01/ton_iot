from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel

EventType = Literal[
    "RUN_STARTED", "PIPELINE_GRAPH_READY", "TASK_STARTED", "LOG_LINE",
    "RESOURCE_SNAPSHOT", "PROGRESS", "ARTIFACT_CREATED",
    "TABLE_PROFILE_READY", "DISTRIBUTION_READY",
    "TASK_FINISHED", "RUN_FINISHED", "STATUS"
]

class Event(BaseModel):
    type: EventType
    payload: Dict[str, Any]
    ts: float
    run_id: str
    task_name: Optional[str] = None
