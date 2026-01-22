import os
import time
import polars as pl
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.core.contracts.artifacts import AlignmentArtifact
from src.app.pipeline.registry import TaskRegistry

@TaskRegistry.register("T05_AlignFeatures")
class T05_AlignFeatures(Task):
    def run(self, context: DAGContext) -> TaskResult:
        from src.infra.resources.monitor import ResourceMonitor
        monitor = ResourceMonitor(context.event_bus, context.run_id)
        monitor.snapshot(self.name)
        
        start_ts = time.time()
        cic_art = context.artifact_store.load_table("cic_consolidated")
        ton_art = context.artifact_store.load_table("ton_clean")
        
        output_path = os.path.join(context.config.paths.work_dir, "artifacts", "alignment_mapping.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        context.logger.info("alignment", "Aligning features between CIC and TON")
        
        # In a real scenario, we would use statistical tests (Cosine, KS, Wasserstein)
        # Here we implement the logic requested: intersection of numeric columns
        
        cic_cols = set(cic_art.columns)
        ton_cols = set(ton_art.columns)
        
        # Simple intersection for demo, excluding labels and metadata
        exclude = {"y", "Label", "source_file", "sample_id", "type", "ts"}
        common = sorted(list((cic_cols & ton_cols) - exclude))
        
        intersection_size = len(common)
        status = "ok"
        reason = ""

        if intersection_size == 0:
            context.logger.warning("alignment", "No direct intersection found. Attempting fallback.")
            # Fallback: top numeric columns from CIC that might exist in TON after renaming rules
            # For this project, we assume some columns match or we use a predefined list if in test_mode
            if context.config.test_mode:
                # VERIFY: Predefined common features for test mode
                # Option 1: Use a hardcoded list of known common features
                # Option 2: Use top 5 numeric features from CIC as dummy common
                common = [c for c in cic_art.columns if c not in exclude][:5]
                status = "degraded"
                reason = "No alignable features in test sample, using CIC top numeric as fallback"
                context.logger.info("alignment", f"Fallback triggered: {reason}")
            else:
                return TaskResult(
                    task_name=self.name,
                    status="failed",
                    duration_s=time.time() - start_ts,
                    error="No alignable features in test sample"
                )

        context.logger.info("alignment", "Feature alignment complete",
                            intersection_size=intersection_size,
                            F_common=common,
                            status=status,
                            reason=reason)

        # Create a dummy mapping table for the artifact
        mapping_df = pl.DataFrame({
            "feature": common,
            "status": [status] * len(common)
        })
        context.table_io.write_parquet(mapping_df, output_path)

        artifact = AlignmentArtifact(
            artifact_id="alignment_spec",
            mapping_table_path=output_path,
            F_common=common,
            metrics_summary={"intersection_size": intersection_size, "status": status, "reason": reason}
        )
        context.artifact_store.save_alignment(artifact)
        
        monitor.snapshot(self.name)
        return TaskResult(
            task_name=self.name,
            status=status,
            duration_s=time.time() - start_ts,
            outputs=["alignment_spec"]
        )
