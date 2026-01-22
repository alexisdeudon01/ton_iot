import time
from src.app.ui.state import UIState
from src.app.ui.reducer import reduce_event
from src.core.contracts.events import Event

def test_ui_flow_sequence():
    state = UIState()
    
    # 1. Run started
    state = reduce_event(state, Event(
        type="RUN_STARTED",
        payload={"config": {
            "paths": {"cic_dir_path": "a", "ton_csv_path": "b", "work_dir": "c", "artifacts_dir": "d"},
            "io": {}, "sampling_policy": {}, "alignment": {}, "preprocessing": {}, "clustering": {},
            "training": {"algorithms": ["LR", "DT", "RF", "CNN", "TabNet"]}, "fusion": {}
        }},
        ts=time.time(),
        run_id="run123"
    ))
    assert state.is_running is True
    assert state.run_id == "run123"
    
    # 2. Task started
    state = reduce_event(state, Event(
        type="TASK_STARTED",
        payload={"inputs": []},
        ts=time.time(),
        run_id="run123",
        task_name="T01"
    ))
    assert state.task_status["T01"].status == "running"
    
    # 3. Log line
    state = reduce_event(state, Event(
        type="LOG_LINE",
        payload={"action": "load", "message": "loading data"},
        ts=time.time(),
        run_id="run123"
    ))
    assert len(state.logs) == 1
    
    # 4. Task finished
    state = reduce_event(state, Event(
        type="TASK_FINISHED",
        payload={"status": "ok", "duration_s": 1.5, "task_name": "T01"},
        ts=time.time(),
        run_id="run123",
        task_name="T01"
    ))
    assert state.task_status["T01"].status == "ok"
    assert state.task_status["T01"].duration == 1.5
