import polars as pl
import os
import time
import psutil
from typing import List, Dict, Any
from src.core.dag.engine import Task, TaskContext
from src.core.contracts.artifacts import TableArtifact, TableProfile, DatasetSpec, AlignmentArtifact
from src.core.events.models import PipelineEvent, EventType

def emit_resource_snapshot(ctx: TaskContext, task_name: str):
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    snapshot = {
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_used_mb": mem.rss / (1024 * 1024),
        "ram_avail_mb": psutil.virtual_memory().available / (1024 * 1024),
        "process_rss_mb": mem.rss / (1024 * 1024),
        "peak_rss_mb": getattr(process.memory_info(), 'peak_wset', 0) / (1024 * 1024) # Windows specific, fallback to 0
    }
    ctx.event_bus.publish(PipelineEvent(
        type=EventType.RESOURCE_SNAPSHOT,
        run_id=ctx.run_id,
        task_name=task_name,
        payload=snapshot
    ))

class T00_InitRun(Task):
    def run(self, ctx: TaskContext) -> List[str]:
        emit_resource_snapshot(ctx, self.name)
        ctx.logger.log("INFO", f"Initializing run {ctx.run_id}")
        # Generate Mermaid graph (simplified for now)
        mmd = "graph TD\n"
        mmd += "  T00[InitRun] --> T01[ConsolidateCIC]\n"
        mmd += "  T00 --> T02[CleanTON]\n"
        mmd += "  T01 --> T03[ProfileCIC]\n"
        mmd += "  T02 --> T04[ProfileTON]\n"
        mmd += "  T03 --> T05[AlignFeatures]\n"
        mmd += "  T04 --> T05\n"
        
        mmd_path = os.path.join(ctx.config.work_dir, "pipeline_graph.mmd")
        os.makedirs(ctx.config.work_dir, exist_ok=True)
        with open(mmd_path, "w") as f:
            f.write(mmd)
            
        ctx.event_bus.publish(PipelineEvent(
            type=EventType.PIPELINE_GRAPH_READY,
            run_id=ctx.run_id,
            payload={"mmd_path": mmd_path, "mmd_text": mmd}
        ))
        emit_resource_snapshot(ctx, self.name)
        return []

class T01_ConsolidateCIC(Task):
    def run(self, ctx: TaskContext) -> List[str]:
        emit_resource_snapshot(ctx, self.name)
        cic_dir = ctx.config.cic_dir_path
        csv_files = [os.path.join(cic_dir, f) for f in os.listdir(cic_dir) if f.endswith(".csv")]
        
        lfs = []
        for f in csv_files:
            lf = ctx.io.read_table(f).with_columns([
                pl.lit(os.path.basename(f)).alias("source_file")
            ])
            # Drop Unnamed: 0 if exists
            if "Unnamed: 0" in lf.columns:
                lf = lf.drop("Unnamed: 0")
            lfs.append(lf)
            
        combined = pl.concat(lfs)
        
        # Label rule: y=0 if Label == "BENIGN", else 1
        combined = combined.with_columns([
            pl.when(pl.col("Label") == "BENIGN").then(0).otherwise(1).alias("y")
        ])
        
        out_path = os.path.join(ctx.config.work_dir, "cic_consolidated.parquet")
        artifact = ctx.io.write_table(combined, out_path)
        artifact.source_step = self.name
        ctx.artifacts.save_artifact("cic_consolidated", artifact)
        
        ctx.event_bus.publish(PipelineEvent(
            type=EventType.ARTIFACT_CREATED,
            run_id=ctx.run_id,
            task_name=self.name,
            payload={"artifact": artifact.model_dump()}
        ))
        emit_resource_snapshot(ctx, self.name)
        return ["cic_consolidated"]

class T02_CleanTON(Task):
    def run(self, ctx: TaskContext) -> List[str]:
        emit_resource_snapshot(ctx, self.name)
        ton_path = ctx.config.ton_csv_path
        lf = ctx.io.read_table(ton_path).with_columns([
            pl.lit(os.path.basename(ton_path)).alias("source_file")
        ])
        
        # Filter type in {"normal", "ddos"}
        lf = lf.filter(pl.col("type").is_in(["normal", "ddos"]))
        
        # Label rule: y=1 if type=="ddos" else 0
        lf = lf.with_columns([
            pl.when(pl.col("type") == "ddos").then(1).otherwise(0).alias("y")
        ])
        
        out_path = os.path.join(ctx.config.work_dir, "ton_clean.parquet")
        artifact = ctx.io.write_table(lf, out_path)
        artifact.source_step = self.name
        ctx.artifacts.save_artifact("ton_clean", artifact)
        
        ctx.event_bus.publish(PipelineEvent(
            type=EventType.ARTIFACT_CREATED,
            run_id=ctx.run_id,
            task_name=self.name,
            payload={"artifact": artifact.model_dump()}
        ))
        emit_resource_snapshot(ctx, self.name)
        return ["ton_clean"]

class T03_ProfileCIC(Task):
    def run(self, ctx: TaskContext) -> List[str]:
        emit_resource_snapshot(ctx, self.name)
        art = TableArtifact(**ctx.artifacts.load_artifact("cic_consolidated"))
        lf = ctx.io.read_table(art.path)
        profile = ctx.profiler.profile_table(lf, "cic", self.name)
        profile.dataset = "cic"
        
        ctx.artifacts.save_artifact("cic_profile", profile)
        ctx.event_bus.publish(PipelineEvent(
            type=EventType.TABLE_PROFILE_READY,
            run_id=ctx.run_id,
            task_name=self.name,
            payload={"artifact_id": "cic_consolidated", "profile": profile.model_dump()}
        ))
        emit_resource_snapshot(ctx, self.name)
        return ["cic_profile"]

class T04_ProfileTON(Task):
    def run(self, ctx: TaskContext) -> List[str]:
        emit_resource_snapshot(ctx, self.name)
        art = TableArtifact(**ctx.artifacts.load_artifact("ton_clean"))
        lf = ctx.io.read_table(art.path)
        profile = ctx.profiler.profile_table(lf, "ton", self.name)
        profile.dataset = "ton"
        
        ctx.artifacts.save_artifact("ton_profile", profile)
        ctx.event_bus.publish(PipelineEvent(
            type=EventType.TABLE_PROFILE_READY,
            run_id=ctx.run_id,
            task_name=self.name,
            payload={"artifact_id": "ton_clean", "profile": profile.model_dump()}
        ))
        emit_resource_snapshot(ctx, self.name)
        return ["ton_profile"]

class T05_AlignFeatures(Task):
    def run(self, ctx: TaskContext) -> List[str]:
        emit_resource_snapshot(ctx, self.name)
        # Simplified alignment: find common columns (case-insensitive match)
        cic_art = TableArtifact(**ctx.artifacts.load_artifact("cic_consolidated"))
        ton_art = TableArtifact(**ctx.artifacts.load_artifact("ton_clean"))
        
        cic_cols = [c.lower() for c in cic_art.columns]
        ton_cols = [c.lower() for c in ton_art.columns]
        
        common = list(set(cic_cols) & set(ton_cols))
        # Ensure 'y' and 'source_file' are kept if present
        if "y" not in common: common.append("y")
        
        align_art = AlignmentArtifact(
            mapping_table=cic_art,  # Placeholder
            F_common=common,
            metrics_summary={"n_common": len(common)},
        )
        ctx.artifacts.save_artifact("alignment", align_art)
        emit_resource_snapshot(ctx, self.name)
        return ["alignment"]


class T06_ProjectCICToCommon(Task):
    def run(self, ctx: TaskContext) -> List[str]:
        emit_resource_snapshot(ctx, self.name)
        align = AlignmentArtifact(**ctx.artifacts.load_artifact("alignment"))
        art = TableArtifact(**ctx.artifacts.load_artifact("cic_consolidated"))
        lf = ctx.io.read_table(art.path).select(align.F_common)

        out_path = os.path.join(ctx.config.work_dir, "cic_projected.parquet")
        artifact = ctx.io.write_table(lf, out_path)
        artifact.source_step = self.name
        ctx.artifacts.save_artifact("cic_projected", artifact)
        emit_resource_snapshot(ctx, self.name)
        return ["cic_projected"]


class T07_ProjectTONToCommon(Task):
    def run(self, ctx: TaskContext) -> List[str]:
        emit_resource_snapshot(ctx, self.name)
        align = AlignmentArtifact(**ctx.artifacts.load_artifact("alignment"))
        art = TableArtifact(**ctx.artifacts.load_artifact("ton_clean"))
        lf = ctx.io.read_table(art.path).select(align.F_common)

        out_path = os.path.join(ctx.config.work_dir, "ton_projected.parquet")
        artifact = ctx.io.write_table(lf, out_path)
        artifact.source_step = self.name
        ctx.artifacts.save_artifact("ton_projected", artifact)
        emit_resource_snapshot(ctx, self.name)
        return ["ton_projected"]


class T08_BuildPreprocessCIC(Task):
    def run(self, ctx: TaskContext) -> List[str]:
        emit_resource_snapshot(ctx, self.name)
        # Simplified: just identify numeric/cat features
        art = TableArtifact(**ctx.artifacts.load_artifact("cic_projected"))
        num_features = [
            c for c, d in art.dtypes.items() if "Float" in d or "Int" in d and c != "y"
        ]
        cat_features = [
            c
            for c, d in art.dtypes.items()
            if "String" in d and c not in ["y", "source_file"]
        ]

        prep = PreprocessArtifact(
            preprocess_path="none",
            num_features=num_features,
            cat_features=cat_features,
            feature_order=num_features + cat_features,
            steps_params={},
            version="1.0.0",
        )
        ctx.artifacts.save_artifact("cic_preprocess", prep)
        emit_resource_snapshot(ctx, self.name)
        return ["cic_preprocess"]


class T12_TuneTrainCIC(Task):
    def run(self, ctx: TaskContext) -> List[str]:
        emit_resource_snapshot(ctx, self.name)
        # Placeholder for training
        ctx.logger.log("INFO", "Training models for CIC...")
        # In a real implementation, we'd use the adapters here
        emit_resource_snapshot(ctx, self.name)
        return []


class T16_LateFusion(Task):
    def run(self, ctx: TaskContext) -> List[str]:
        emit_resource_snapshot(ctx, self.name)
        ctx.logger.log("INFO", "Performing Late Fusion...")
        emit_resource_snapshot(ctx, self.name)
        return []


class T17_Evaluate(Task):
    def run(self, ctx: TaskContext) -> List[str]:
        emit_resource_snapshot(ctx, self.name)
        ctx.logger.log("INFO", "Final Evaluation...")
        emit_resource_snapshot(ctx, self.name)
        return []
