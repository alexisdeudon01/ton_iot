import time
from src.core.dag.graph import DAGGraph
from src.core.dag.runner import DAGRunner
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.infra.events.queue_bus import QueueEventBus
from src.infra.logging.structured_logger import StructuredEventLogger

class DummyTask(Task):
    def run(self, context: DAGContext) -> TaskResult:
        return TaskResult(task_name=self.name, status="ok", duration_s=0.1)

def test_dag_runner_execution():
    graph = DAGGraph()
    graph.add_task(DummyTask("A"))
    graph.add_task(DummyTask("B"), depends_on=["A"])
    
    bus = QueueEventBus()
    logger = StructuredEventLogger(bus, "test")
    
    # Mock context
    context = DAGContext(
        run_id="test",
        config=None, # Not needed for dummy
        artifact_store=None,
        event_bus=bus,
        logger=logger,
        table_io=None
    )
    
    runner = DAGRunner(graph, context)
    results = runner.run()
    
    assert len(results) == 2
    assert results[0].task_name == "A"
    assert results[1].task_name == "B"
    bus.stop()
