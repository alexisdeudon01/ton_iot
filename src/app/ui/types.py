from typing import Dict, Optional
from pydantic import BaseModel

class TaskStatus(BaseModel):
    task_name: str
    status: str = "pending" # pending, running, ok, failed
    started_ts: Optional[float] = None
    ended_ts: Optional[float] = None
    duration: float = 0.0
    meta: dict = {}
