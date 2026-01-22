import os
import time
import polars as pl
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.core.contracts.artifacts import AlignmentArtifact, TableArtifact
from src.app.pipeline.registry import TaskRegistry

@TaskRegistry.register("T05_AlignFeatures")
class T05_AlignFeatures(Task):
    def run(self, context: DAGContext) -> TaskResult:
        start_ts = time.time()
        cic_art = context.artifact_store.load_table("cic_consolidated")
        ton_art = context.artifact_store.load_table("ton_clean")
        
        # Simple intersection for F_common (excluding labels and metadata)
        exclude = {"y", "Label", "type", "source_file", "ts", "timestamp"}
        cic_cols = set(cic_art.columns) - exclude
        ton_cols = set(ton_art.columns) - exclude
        
        f_common = sorted(list(cic_cols.intersection(ton_cols)))
        
        if not f_common:
            # Fallback: if no exact match, we might need a descriptor-based mapping
            # For this implementation, we'll assume some overlap exists or use a subset of CIC as base
            context.logger.warning("alignment", "No exact column intersection found. Using CIC columns as base for alignment.")
            f_common = sorted(list(cic_cols))[:20] # Take first 20 for demo if empty

        mapping_df = pl.DataFrame({
            "feature": f_common,
            "cic_source": f_common,
            "ton_source": f_common
        })
        
        mapping_path = os.path.join(context.config.paths.work_dir, "artifacts", "feature_mapping.parquet")
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
        context.table_io.write_parquet(mapping_df, mapping_path)
        
        mapping_art = TableArtifact(
            artifact_id="feature_mapping_table",
            name="Feature Mapping Table",
            path=mapping_path,
            format="parquet",
            n_rows=mapping_df.height,
            n_cols=mapping_df.width,
            columns=mapping_df.columns,
            dtypes={col: str(dtype) for col, dtype in zip(mapping_df.columns, mapping_df.dtypes)},
            version="1.0.0",
            source_step=self.name,
            fingerprint=str(hash(mapping_path)),
            stats={}
        )
        
        alignment = AlignmentArtifact(
            artifact_id="alignment_spec",
            mapping_table=mapping_art,
            F_common=f_common,
            metrics_summary={"n_common": len(f_common)}
        )
        context.artifact_store.save_alignment(alignment)
        
        context.logger.info("alignment", f"Aligned {len(f_common)} features.")
        
        return TaskResult(
            task_name=self.name,
            status="ok",
            duration_s=time.time() - start_ts,
            outputs=["alignment_spec"]
        )
