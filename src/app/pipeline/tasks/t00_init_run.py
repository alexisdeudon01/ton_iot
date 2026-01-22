import time
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.core.contracts.events import Event
from src.app.pipeline.registry import TaskRegistry

@TaskRegistry.register("T00_InitRun")
class T00_InitRun(Task):
    def run(self, context: DAGContext) -> TaskResult:
        start_ts = time.time()
        
        # Publish RUN_STARTED
        context.event_bus.publish(Event(
            type="RUN_STARTED",
            payload={
                "run_id": context.run_id,
                "config_version": context.config.version,
                "config": context.config.model_dump()
            },
            ts=start_ts,
            run_id=context.run_id
        ))
        
        context.logger.info("init", f"Run {context.run_id} initialized.")
        
        return TaskResult(
            task_name=self.name,
            status="ok",
            duration_s=time.time() - start_ts,
            outputs=[]
        )
