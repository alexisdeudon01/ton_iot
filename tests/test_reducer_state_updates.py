import time
from src.app.ui.state import UIState
from src.app.ui.reducer import reduce_event
from src.core.contracts.events import Event

def test_reducer_task_started():
    state = UIState()
    event = Event(
        type="TASK_STARTED",
        payload={"inputs": []},
        ts=12345.6,
        run_id="run1",
        task_name="TaskA"
    )
    
    new_state = reduce_event(state, event)
    
    assert "TaskA" in new_state.task_status
    assert new_state.task_status["TaskA"].status == "running"
    assert new_state.task_status["TaskA"].started_ts == 12345.6

def test_reducer_log_line():
    state = UIState()
    event = Event(
        type="LOG_LINE",
        payload={"message": "hello", "action": "test"},
        ts=time.time(),
        run_id="run1"
    )
    
    new_state = reduce_event(state, event)
    assert len(new_state.logs) == 1
    assert new_state.logs[0]["message"] == "hello"
