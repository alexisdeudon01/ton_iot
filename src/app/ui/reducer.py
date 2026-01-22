from src.app.ui.state import UIState
from src.core.contracts.events import Event
from src.app.ui.types import TaskStatus
from src.core.contracts.config import PipelineConfig
from src.core.contracts.artifacts import TableArtifact
from src.core.contracts.profiles import TableProfile, DistributionBundle

def reduce_event(state: UIState, event: Event) -> UIState:
    new_state = state.model_copy(deep=True)
    
    if event.type == "RUN_STARTED":
        new_state.run_id = event.run_id
        new_state.config = PipelineConfig.model_validate(event.payload["config"])
        new_state.is_running = True
        new_state.status_message = f"Run {event.run_id} started"
        
    elif event.type == "PIPELINE_GRAPH_READY":
        new_state.pipeline_mmd = event.payload["mermaid"]
        
    elif event.type == "TASK_STARTED":
        task_name = event.task_name
        new_state.task_status[task_name] = TaskStatus(
            task_name=task_name,
            status="running",
            started_ts=event.ts
        )
        new_state.status_message = f"Task {task_name} started"
        
    elif event.type == "LOG_LINE":
        new_state.logs.append(event.payload)
        if len(new_state.logs) > 5000:
            new_state.logs.pop(0)
            
    elif event.type == "RESOURCE_SNAPSHOT":
        new_state.resources.append(event.payload)
        if len(new_state.resources) > 2000:
            new_state.resources.pop(0)
            
    elif event.type == "ARTIFACT_CREATED":
        # payload contains artifact as dict
        # We only track TableArtifacts in the main list for now
        art_data = event.payload.get("artifact")
        if art_data and "n_rows" in art_data:
            new_state.artifacts.append(TableArtifact.model_validate(art_data))
            
    elif event.type == "TABLE_PROFILE_READY":
        profile = TableProfile.model_validate(event.payload["profile"])
        new_state.last_profiles[event.payload["artifact_id"]] = profile
        
    elif event.type == "DISTRIBUTION_READY":
        bundle = DistributionBundle.model_validate(event.payload["bundle"])
        key = f"{bundle.artifact_id}:{bundle.feature}"
        new_state.last_distributions[key] = bundle
        
    elif event.type == "TASK_FINISHED":
        task_name = event.task_name
        if task_name in new_state.task_status:
            ts = new_state.task_status[task_name]
            ts.status = event.payload["status"]
            ts.ended_ts = event.ts
            ts.duration = event.payload["duration_s"]
            ts.meta = event.payload.get("meta", {})
        new_state.status_message = f"Task {task_name} finished ({event.payload['status']})"
            
    elif event.type == "RUN_FINISHED":
        new_state.is_running = False
        new_state.status_message = f"Run finished: {event.payload['status']}"
        
    elif event.type == "STATUS":
        new_state.status_message = event.payload["message"]
        
    return new_state
