from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from src.core.contracts.config import PipelineConfig
from src.core.contracts.artifacts import TableArtifact
from src.core.contracts.profiles import TableProfile, DistributionBundle
from src.app.ui.types import TaskStatus

class UIState(BaseModel):
    run_id: Optional[str] = None
    config: Optional[PipelineConfig] = None
    logs: List[Dict] = Field(default_factory=list) # Max 5000
    resources: List[Dict] = Field(default_factory=list) # Max 2000
    task_status: Dict[str, TaskStatus] = Field(default_factory=dict)
    artifacts: List[TableArtifact] = Field(default_factory=list)
    last_profiles: Dict[str, TableProfile] = Field(default_factory=dict) # artifact_id -> profile
    last_distributions: Dict[str, DistributionBundle] = Field(default_factory=dict) # "artifact_id:feature" -> bundle
    pipeline_mmd: str = ""
    status_message: str = "Ready"
    is_running: bool = False
