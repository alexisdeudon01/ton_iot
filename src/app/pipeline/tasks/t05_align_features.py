import os
import time
import polars as pl
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.core.contracts.artifacts import AlignmentArtifact, TableArtifact
from src.app.pipeline.registry import TaskRegistry
from src.app.pipeline.universal_feature_mapping import (
    UNIVERSAL_FEATURES,
    CIC_REQUIRED_COLUMNS,
    TON_REQUIRED_COLUMNS,
    mapping_rows,
)


@TaskRegistry.register("T05_AlignFeatures")
class T05_AlignFeatures(Task):
    """Align CIC and TON feature names into a common schema."""

    def run(self, context: DAGContext) -> TaskResult:
        """Compute shared feature names and emit an alignment artifact."""
        from src.infra.resources.monitor import ResourceMonitor

        monitor = ResourceMonitor(context.event_bus, context.run_id)
        monitor.snapshot(self.name)

        start_ts = time.time()
        cic_art = context.artifact_store.load_table("cic_consolidated")
        ton_art = context.artifact_store.load_table("ton_clean")

        output_path = os.path.join(
            context.config.paths.work_dir, "artifacts", "alignment_mapping.parquet"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        context.logger.info("alignment", "Aligning features between CIC and TON")

        cic_cols = set(cic_art.columns)
        ton_cols = set(ton_art.columns)
        exclude = {"y", "Label", "source_file", "sample_id", "type", "ts"}
        raw_common = sorted(list((cic_cols & ton_cols) - exclude))

        intersection_size = len(raw_common)
        common = raw_common
        status = "ok"
        reason = ""
        source = "intersection"

        context.logger.debug(
            "alignment",
            "Alignment input summary",
            cic_columns_count=len(cic_cols),
            ton_columns_count=len(ton_cols),
            excluded_columns=sorted(exclude),
            raw_intersection_count=intersection_size,
        )

        missing_cic = sorted([c for c in CIC_REQUIRED_COLUMNS if c not in cic_cols])
        missing_ton = sorted([c for c in TON_REQUIRED_COLUMNS if c not in ton_cols])
        if missing_cic or missing_ton:
            reason = "Missing required columns for universal feature mapping"
            context.logger.warning(
                "alignment",
                reason,
                missing_cic=missing_cic,
                missing_ton=missing_ton,
            )
            return TaskResult(
                task_name=self.name,
                status="failed",
                duration_s=time.time() - start_ts,
                error=f"{reason}: CIC={missing_cic} TON={missing_ton}",
            )

        common = list(UNIVERSAL_FEATURES)
        status = "ok"
        reason = "Manual universal feature mapping"
        source = "manual"

        rows = []
        for row in mapping_rows():
            rows.append({**row, "status": status, "source": source, "reason": reason})

        mapping_df = pl.DataFrame(rows)
        context.table_io.write_parquet(mapping_df, output_path)

        mapping_art = TableArtifact(
            artifact_id="alignment_mapping_table",
            name="Alignment Mapping",
            path=output_path,
            format="parquet",
            n_rows=mapping_df.height,
            n_cols=mapping_df.width,
            columns=mapping_df.columns,
            dtypes={c: str(t) for c, t in zip(mapping_df.columns, mapping_df.dtypes)},
            version="1.0.0",
            source_step=self.name,
            fingerprint=str(hash(output_path)),
            stats={},
        )

        artifact = AlignmentArtifact(
            artifact_id="alignment_spec",
            mapping_table=mapping_art,
            F_common=common,
            metrics_summary={
                "raw_intersection_size": intersection_size,
                "selected_feature_count": len(common),
                "status": status,
                "reason": reason,
            },
        )
        context.artifact_store.save_alignment(artifact)

        context.logger.info(
            "alignment",
            "Feature alignment complete",
            intersection_size=intersection_size,
            F_common_count=len(common),
            status=status,
        )

        monitor.snapshot(self.name)
        return TaskResult(
            task_name=self.name,
            status=status,
            duration_s=time.time() - start_ts,
            outputs=["alignment_spec"],
        )
