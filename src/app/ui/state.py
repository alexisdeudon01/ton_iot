from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from src.core.events.models import PipelineEvent

class AppState(BaseModel):
    logs: List[str] = Field(default_factory=list)
    active_task: Optional[str] = None
    task_status: Dict[str, str] = Field(default_factory=dict) # task_name -> status
    task_durations: Dict[str, float] = Field(default_factory=dict)
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    last_profiles: Dict[str, Dict[str, Any]] = Field(default_factory=dict) # artifact_id -> profile
    last_distributions: Dict[str, Dict[str, Any]] = Field(default_factory=dict) # artifact_id:feature -> dist
    resources: List[Dict[str, Any]] = Field(default_factory=list)
    peak_rss_mb: float = 0.0
    pipeline_mmd: str = ""
    status_message: str = "Ready"
    config: Optional[Dict[str, Any]] = None
    run_id: Optional[str] = None
