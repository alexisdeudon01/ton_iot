import time
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.core.contracts.events import Event
from src.app.pipeline.registry import TaskRegistry
from src.infra.profiling.profiler import PolarsProfiler

@TaskRegistry.register("T04_ProfileTON")
class T04_ProfileTON(Task):
    def run(self, context: DAGContext) -> TaskResult:
        start_ts = time.time()
        artifact = context.artifact_store.load_table("ton_clean")
        lf = context.table_io.read_parquet(artifact.path)
        
        profiler = PolarsProfiler()
        profile = profiler.profile_table(lf, artifact.artifact_id, "ton", self.name)
        
        context.event_bus.publish(Event(
            type="TABLE_PROFILE_READY",
            payload={"artifact_id": artifact.artifact_id, "profile": profile.model_dump()},
            ts=time.time(),
            run_id=context.run_id,
            task_name=self.name
        ))
        
        context.logger.info("profiling", f"Profiled TON: {profile.n_rows} rows")
        
        return TaskResult(
            task_name=self.name,
            status="ok",
            duration_s=time.time() - start_ts,
            outputs=[f"profile_{artifact.artifact_id}"]
        )
