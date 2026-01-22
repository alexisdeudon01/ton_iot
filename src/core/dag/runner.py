import time
import traceback
from typing import Dict, List
from src.core.dag.graph import DAGGraph
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.core.contracts.events import Event

class DAGRunner:
    def __init__(self, graph: DAGGraph, context: DAGContext):
        self.graph = graph
        self.context = context
        self.results: Dict[str, TaskResult] = {}

    def run(self) -> List[TaskResult]:
        order = self.graph.get_execution_order()
        
        # Publish graph ready
        self.context.event_bus.publish(Event(
            type="PIPELINE_GRAPH_READY",
            payload={"mermaid": self.graph.to_mermaid()},
            ts=time.time(),
            run_id=self.context.run_id
        ))

        for task_name in order:
            task = self.graph.tasks[task_name]
            start_ts = time.time()
            
            self.context.event_bus.publish(Event(
                type="TASK_STARTED",
                payload={"inputs": task.inputs},
                ts=start_ts,
                run_id=self.context.run_id,
                task_name=task_name
            ))

            try:
                result = task.run(self.context)
                self.results[task_name] = result
            except Exception as e:
                duration = time.time() - start_ts
                result = TaskResult(
                    task_name=task_name,
                    status="failed",
                    duration_s=duration,
                    error=str(e),
                    meta={"traceback": traceback.format_exc()}
                )
                self.results[task_name] = result
                
                self.context.event_bus.publish(Event(
                    type="TASK_FINISHED",
                    payload=result.model_dump(),
                    ts=time.time(),
                    run_id=self.context.run_id,
                    task_name=task_name
                ) )
                break # Stop execution on failure

            self.context.event_bus.publish(Event(
                type="TASK_FINISHED",
                payload=result.model_dump(),
                ts=time.time(),
                run_id=self.context.run_id,
                task_name=task_name
            ))

        return list(self.results.values())
