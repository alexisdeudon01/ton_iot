from typing import Optional, List, Any
from pydantic import BaseModel

class TaskResult(BaseModel):
    task_name: str
    status: str  # "ok", "failed", "skipped"
    duration_s: float
    outputs: List[str] = []
    error: Optional[str] = None
    meta: dict = {}
