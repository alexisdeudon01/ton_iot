from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Any

from pydantic import BaseModel, Field, model_validator
import matplotlib

matplotlib.use('Agg')

FusionMode = Literal["late"]
DatasetName = Literal["cic", "ton"]
ModelName = Literal["LR", "DT", "RF", "CNN", "TabNet"]
TuningMode = Literal["none", "grid", "random"]
ScalerName = Literal["robust"]
SelectorName = Literal["mutual_info", "f_classif"]
ParquetCompression = Literal["snappy", "zstd", "gzip", "brotli"]


class PathsConfig(BaseModel):
    # Inputs
    cic_dir: Path = Field(default=Path("/home/tor/ton_iot/ton_iot/datasets/cic_ddos2019"))
    ton_file: Path = Field(default=Path("/home/tor/ton_iot/ton_iot/datasets/ton_iot/train_test_network.csv"))

    # Outputs
    out_root: Path = Field(default=Path("output"))
    data_parquet_root: Path = Field(default=Path("output/data_parquet"))
    artifacts_root: Path = Field(default=Path("output/artifacts"))
    reports_root: Path = Field(default=Path("output/reports"))
    logs_root: Path = Field(default=Path("output/logs"))

    # Prepared parquet datasets
    cic_parquet: Path = Field(default=Path("output/data_parquet/cic"))
    ton_parquet: Path = Field(default=Path("output/data_parquet/ton"))

    # Logs
    prepare_log: Path = Field(default=Path("output/logs/prepare_datasets.log"))
    run_log: Path = Field(default=Path("output/logs/pipeline_run.log"))

    @model_validator(mode="after")
    def _sync_paths(self) -> "PathsConfig":
        out = self.out_root
        self.data_parquet_root = out / "data_parquet"
        self.artifacts_root = out / "artifacts"
        self.reports_root = out / "reports"
        self.logs_root = out / "logs"

        self.cic_parquet = self.data_parquet_root / "cic"
        self.ton_parquet = self.data_parquet_root / "ton"

        self.prepare_log = self.logs_root / "prepare_datasets.log"
        self.run_log = self.logs_root / "pipeline_run.log"
        return self


class Phase0PrepareConfig(BaseModel):
    chunksize: int = Field(default=250_000, ge=10_000)
    low_memory: bool = True
    parquet_compression: ParquetCompression = "snappy"
    partition_by_label: bool = False # Set to False for simplicity in this pipeline
    cic_label_col: str = "Label"
    cic_benign_value: str = "BENIGN"
    cic_add_source_col: bool = True
    cic_source_col: str = "source"
    ton_type_col: str = "type"
    ton_allowed_types: Tuple[str, str] = ("normal", "ddos")
    label_col: str = "label"
    drop_columns_if_present: List[str] = Field(default_factory=lambda: ["Unnamed: 0"])
    export_csv_debug: bool = False
    debug_csv_dir: Optional[Path] = None


class Phase1AlignmentConfig(BaseModel):
    align_sample_rows: int = Field(default=100_000, ge=10_000)
    stratified_sample: bool = True
    random_state: int = 42
    numeric_only: bool = True
    drop_non_informative: bool = True
    max_missing_ratio: float = Field(default=0.3, ge=0.0, le=1.0)
    enable_string_similarity: bool = True
    string_sim_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    enable_descriptor_similarity: bool = True
    descriptor_sim_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    enable_ks_test: bool = True
    ks_pvalue_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    enable_wasserstein: bool = True
    wasserstein_max: Optional[float] = None
    alignment_artifacts_dir: Path = Field(default=Path("output/artifacts/feature_alignment"))
    mapping_pairs_csv: str = "mapping_pairs.csv"
    alignment_metrics_csv: str = "alignment_metrics.csv"
    f_common_json: str = "f_common.json"


class Phase2PreprocessConfig(BaseModel):
    max_missing_ratio: float = Field(default=0.3, ge=0.0, le=1.0)
    scaler: ScalerName = "robust"
    impute_numeric_strategy: Literal["median"] = "median"
    test_size: float = Field(default=0.20, ge=0.05, le=0.5)
    val_size: float = Field(default=0.20, ge=0.05, le=0.5)
    stratify: bool = True
    random_state: int = 42
    splits_root: Path = Field(default=Path("output/data_parquet/splits"))
    split_manifest_json: str = "split_manifest.json"


class Phase3SelectionConfig(BaseModel):
    selector_primary: SelectorName = "mutual_info"
    selector_secondary: SelectorName = "f_classif"
    top_k: int = Field(default=40, ge=5)
    min_fusion_features: int = Field(default=10, ge=1)
    selection_artifacts_dir: Path = Field(default=Path("output/artifacts/feature_selection"))
    f_fusion_json: str = "f_fusion.json"


class TuningConfig(BaseModel):
    mode: TuningMode = "random"
    cv_folds: int = Field(default=5, ge=2, le=10)
    n_iter_random: int = Field(default=25, ge=1)
    primary_metric: Literal["recall", "f1", "roc_auc"] = "recall"
    grids: Dict[ModelName, Dict[str, List]] = Field(
        default_factory=lambda: {
            "LR": {"C": [0.1, 1.0, 10.0], "max_iter": [1000]},
            "DT": {"max_depth": [5, 10, 20, None], "min_samples_split": [2, 10, 50]},
            "RF": {"n_estimators": [100, 300], "max_depth": [None, 20], "min_samples_split": [2, 10]},
            "CNN": {"epochs": [10, 20], "batch_size": [256, 512], "lr": [1e-3, 3e-4]},
            "TabNet": {"n_d": [8, 16], "n_a": [8, 16], "n_steps": [3, 5], "lr": [2e-2, 1e-2]},
        }
    )


class Phase4TrainingConfig(BaseModel):
    model_cic: ModelName = "RF"
    model_ton: ModelName = "TabNet"
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    calibrate_probs: bool = True
    calibration_method: Literal["platt", "isotonic"] = "platt"
    random_state: int = 42
    models_root: Path = Field(default=Path("output/artifacts/models"))


class Phase5LateFusionConfig(BaseModel):
    fusion_mode: FusionMode = "late"
    weight_grid_step: float = Field(default=0.05, ge=0.01, le=0.5)
    optimize_metric: Literal["recall", "f1", "roc_auc"] = "recall"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    optimize_threshold: bool = False
    fusion_artifacts_dir: Path = Field(default=Path("output/artifacts/models/fusion"))
    fusion_config_json: str = "fusion_config.json"
    fusion_curve_csv: str = "fusion_curve.csv"


class Phase6EvalConfig(BaseModel):
    metrics: List[str] = Field(default_factory=lambda: ["recall", "f1", "roc_auc", "recall_at_k"])
    recall_at_k: List[int] = Field(default_factory=lambda: [100, 500, 1000])
    enable_transfer_tests: bool = True


class Phase7ResourcesConfig(BaseModel):
    measure_peak_memory_process: bool = True
    measure_cpu_process: bool = True
    measure_latency: bool = True
    measure_dask_workers: bool = False
    latency_samples: int = Field(default=10_000, ge=100)


class Phase8XAIConfig(BaseModel):
    enable_xai: bool = True
    enable_permutation_importance: bool = True
    enable_lime: bool = True
    enable_shap: bool = True
    xai_sample_rows: int = Field(default=5_000, ge=100)
    random_state: int = 42
    xai_artifacts_dir: Path = Field(default=Path("output/artifacts/xai"))


class ReportConfig(BaseModel):
    report_csv: Path = Field(default=Path("output/reports/final_report.csv"))
    report_md: Path = Field(default=Path("output/reports/final_report.md"))
    columns: List[str] = Field(
        default_factory=lambda: [
            "model_name",
            "dataset",
            "detection_performance",
            "resource_efficiency",
            "explainability",
            "label_balance",
            "notes",
        ]
    )


class PipelineConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    phase0: Phase0PrepareConfig = Field(default_factory=Phase0PrepareConfig)
    phase1: Phase1AlignmentConfig = Field(default_factory=Phase1AlignmentConfig)
    phase2: Phase2PreprocessConfig = Field(default_factory=Phase2PreprocessConfig)
    phase3: Phase3SelectionConfig = Field(default_factory=Phase3SelectionConfig)
    phase4: Phase4TrainingConfig = Field(default_factory=Phase4TrainingConfig)
    phase5: Phase5LateFusionConfig = Field(default_factory=Phase5LateFusionConfig)
    phase6: Phase6EvalConfig = Field(default_factory=Phase6EvalConfig)
    phase7: Phase7ResourcesConfig = Field(default_factory=Phase7ResourcesConfig)
    phase8: Phase8XAIConfig = Field(default_factory=Phase8XAIConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    test_mode: bool = False
    verbose_logs: bool = True

    @model_validator(mode="after")
    def _validate_contracts(self) -> "PipelineConfig":
        if self.phase0.label_col != "label":
            raise ValueError("Contract violation: label column must be exactly 'label'.")
        if self.phase5.fusion_mode != "late":
            raise ValueError("This configuration enforces late fusion only.")
        if self.phase2.scaler != "robust":
            raise ValueError("Contract violation: scaler must be RobustScaler ('robust').")
        out = self.paths.out_root
        for p in [
            self.paths.data_parquet_root,
            self.paths.artifacts_root,
            self.paths.reports_root,
            self.paths.logs_root,
        ]:
            if not str(p).startswith(str(out)):
                raise ValueError(f"Output path must be under out_root: {p}")
        return self

config = PipelineConfig()
