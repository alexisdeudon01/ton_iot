from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
import time
import traceback
from pydantic import BaseModel
from src.core.contracts.config import PipelineConfig
from src.core.ports.interfaces import IEventBus, ITableIO, IProfiler, IArtifactStore, ILogger
from src.core.events.models import PipelineEvent, EventType

class TaskResult(BaseModel):
    task_name: str
    status: str # "ok", "failed", "skipped"
    duration_s: float
    outputs: List[str] = []
    error: Optional[str] = None

class TaskContext:
    def __init__(
        self,
        config: PipelineConfig,
        io: ITableIO,
        profiler: IProfiler,
        artifacts: IArtifactStore,
        event_bus: IEventBus,
        logger: ILogger,
        run_id: str
    ):
        self.config = config
        self.io = io
        self.profiler = profiler
        self.artifacts = artifacts
        self.event_bus = event_bus
        self.logger = logger
        self.run_id = run_id
        self.shared_data: Dict[str, Any] = {}

class Task(ABC):
    def __init__(self, name: str, requires: List[str] = None):
        self.name = name
        self.requires = requires or []

    @abstractmethod
    def run(self, ctx: TaskContext) -> List[str]:
        """Returns list of artifact IDs created"""
        pass

    def execute(self, ctx: TaskContext) -> TaskResult:
        start_ts = time.time()
        ctx.event_bus.publish(PipelineEvent(
            type=EventType.TASK_STARTED,
            run_id=ctx.run_id,
            task_name=self.name,
            payload={"requires": self.requires}
        ))
        
        try:
            outputs = self.run(ctx)
            duration = time.time() - start_ts
            result = TaskResult(task_name=self.name, status="ok", duration_s=duration, outputs=outputs)
            
            ctx.event_bus.publish(PipelineEvent(
                type=EventType.TASK_FINISHED,
                run_id=ctx.run_id,
                task_name=self.name,
                payload=result.model_dump()
            ))
            return result
        except Exception as e:
            duration = time.time() - start_ts
            error_msg = traceback.format_exc()
            ctx.logger.log("ERROR", f"Task {self.name} failed: {str(e)}")
            result = TaskResult(task_name=self.name, status="failed", duration_s=duration, error=error_msg)
            
            ctx.event_bus.publish(PipelineEvent(
                type=EventType.TASK_FINISHED,
                run_id=ctx.run_id,
                task_name=self.name,
                payload=result.model_dump()
            ))
            raise e

class DAGRunner:
    def __init__(self, ctx: TaskContext):
        self.ctx = ctx
        self.tasks: Dict[str, Task] = {}

    def add_task(self, task: Task):
        self.tasks[task.name] = task

    def run_all(self):
        self.ctx.event_bus.publish(PipelineEvent(
            type=EventType.RUN_STARTED,
            run_id=self.ctx.run_id,
            payload={"config": self.ctx.config.model_dump()}
        ))
        
        start_ts = time.time()
        executed: Set[str] = set()
        results: List[TaskResult] = []
        
        # Simple topological sort / dependency resolution
        while len(executed) < len(self.tasks):
            ready_tasks = [
                t for name, t in self.tasks.items() 
                if name not in executed and all(req in executed for req in t.requires)
            ]
            
            if not ready_tasks:
                remaining = [t for t in self.tasks if t not in executed]
                raise Exception(f"Circular dependency or missing requirement for tasks: {remaining}")
            
            for task in ready_tasks:
                res = task.execute(self.ctx)
                results.append(res)
                executed.add(task.name)
        
        total_duration = time.time() - start_ts
        self.ctx.event_bus.publish(PipelineEvent(
            type=EventType.RUN_FINISHED,
            run_id=self.ctx.run_id,
            payload={
                "status": "ok",
                "total_time_s": total_duration,
                "tasks_count": len(results)
            }
        ))
