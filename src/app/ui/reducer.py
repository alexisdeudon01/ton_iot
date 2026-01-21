from src.app.ui.state import AppState
from src.core.events.models import PipelineEvent, EventType

def reduce_event(state: AppState, event: PipelineEvent) -> AppState:
    # Create a copy of the state to maintain purity (though Pydantic models are mutable)
    # For performance in Tkinter, we might modify in place, but let's follow the pattern
    
    if event.type == EventType.RUN_STARTED:
        state.run_id = event.run_id
        state.config = event.payload.get("config")
        state.status_message = f"Run {event.run_id} started"
        state.task_status = {}
        state.artifacts = []
        
    elif event.type == EventType.TASK_STARTED:
        state.active_task = event.task_name
        state.task_status[event.task_name] = "running"
        state.status_message = f"Task {event.task_name} in progress..."
        
    elif event.type == EventType.TASK_FINISHED:
        state.active_task = None
        status = event.payload.get("status", "unknown")
        state.task_status[event.task_name] = status
        state.task_durations[event.task_name] = event.payload.get("duration_s", 0.0)
        state.status_message = f"Task {event.task_name} finished with status: {status}"
        
    elif event.type == EventType.LOG_LINE:
        line = f"[{event.task_name or 'GLOBAL'}] {event.payload.get('line')}"
        state.logs.append(line)
        if len(state.logs) > 1000:
            state.logs.pop(0)
            
    elif event.type == EventType.ARTIFACT_CREATED:
        state.artifacts.append(event.payload.get("artifact"))
        
    elif event.type == EventType.TABLE_PROFILE_READY:
        art_id = event.payload.get("artifact_id")
        state.last_profiles[art_id] = event.payload.get("profile")
        
    elif event.type == EventType.DISTRIBUTION_READY:
        art_id = event.payload.get("artifact_id")
        feature = event.payload.get("feature")
        state.last_distributions[f"{art_id}:{feature}"] = event.payload.get("dist")
        
    elif event.type == EventType.RESOURCE_SNAPSHOT:
        state.resources.append(event.payload)
        if len(state.resources) > 100:
            state.resources.pop(0)
        rss = event.payload.get("process_rss_mb", 0.0)
        if rss > state.peak_rss_mb:
            state.peak_rss_mb = rss
            
    elif event.type == EventType.PIPELINE_GRAPH_READY:
        state.pipeline_mmd = event.payload.get("mmd_text", "")
        
    elif event.type == EventType.STATUS:
        state.status_message = event.payload.get("message", "")
        
    return state
