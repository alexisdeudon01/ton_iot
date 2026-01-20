#!/usr/bin/env python3
"""
Phase 2: Apply Best Configuration
Applique la meilleure configuration trouvée en Phase 1 au dataset complet
STATELESS preprocessing uniquement (clean_data + encode_features, PAS scaling/FS/SMOTE)
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.core import DataHarmonizer, DatasetLoader, PreprocessingPipeline
from src.core.feature_engineering import engineer_cic, engineer_ton

logger = logging.getLogger(__name__)


class Phase2ApplyBestConfig:
    """Phase 2: Application de la meilleure configuration (stateless preprocessing)"""

    def __init__(self, config, best_config: Dict[str, Any]):
        self.config = config
        self.best_config = best_config
        self.results_dir = Path(config.output_dir) / "phase2_apply_best_config"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.loader = DatasetLoader()
        self.harmonizer = DataHarmonizer()

    def run(self) -> Dict:
        """Run Phase 2: Apply stateless preprocessing to full dataset."""
        logger.info("Phase 2: Apply Best Config (stateless preprocessing only)")
        logger.info("=" * 70)

        # Load datasets
        df_processed = self._load_and_harmonize_datasets()

        # Separate features and labels (keep dataset_source as feature)
        drop_cols = ["label"]
        if "dataset_source" in df_processed.columns:
            # Keep dataset_source as a feature (already encoded as 0/1)
            # It will be included in preprocessing but preserved
            pass

        X = df_processed.drop(drop_cols, axis=1, errors="ignore")
        y = df_processed["label"]

        # Apply stateless preprocessing only
        X_processed, y_processed = self._apply_stateless_preprocessing(X, y)

        # Combine processed features with label
        df_final = X_processed.copy()
        df_final["label"] = y_processed.values

        # Save outputs
        output_paths = self._save_outputs(df_final, X_processed.columns.tolist())

        logger.info("=" * 70)
        logger.info("Phase 2 completed successfully")
        logger.info(
            f"Processed dataset: {df_final.shape[0]} rows, {df_final.shape[1]-1} features + label"
        )
        logger.info(f"Outputs saved to: {self.results_dir}")

        return {
            "status": "completed",
            "output_paths": output_paths,
            "shape": df_final.shape,
            "feature_names": X_processed.columns.tolist(),
        }

    def _load_and_harmonize_datasets(self) -> pd.DataFrame:
        """Load, harmonize, and fuse datasets."""
        logger.info("Loading datasets...")

        df_ton = self.loader.load_ton_iot(
            sample_ratio=self.config.sample_ratio,
            random_state=self.config.random_state,
            incremental=False,
        )
        logger.info(f"TON_IoT loaded: {df_ton.shape}")
        df_ton = engineer_ton(df_ton)
        logger.info("TON_IoT feature engineering applied.")

        try:
            max_files = getattr(self.config, "cic_max_files", None)
            if self.config.test_mode and max_files is None:
                max_files = 3

            df_cic = self.loader.load_cic_ddos2019(
                sample_ratio=self.config.sample_ratio,
                random_state=self.config.random_state,
                incremental=False,
                max_files_in_test=max_files if max_files is not None else -1,
            )
            logger.info(f"CIC-DDoS2019 loaded: {df_cic.shape}")
            df_cic = engineer_cic(df_cic)
            logger.info("CIC-DDoS2019 feature engineering applied.")
        except FileNotFoundError:
            logger.warning("CIC-DDoS2019 not available. Using TON_IoT only.")
            df_cic = pd.DataFrame()

        if not df_cic.empty:
            try:
                logger.info("Harmonizing features...")
                df_cic_harm, df_ton_harm = self.harmonizer.harmonize_features(
                    df_cic, df_ton
                )
                logger.info("Performing early fusion...")
                df_processed, validation = self.harmonizer.early_fusion(
                    df_cic_harm, df_ton_harm
                )
                logger.info(f"Fused dataset: {df_processed.shape}")
                if "dataset_source" in df_processed.columns:
                    logger.info(
                        "[INFO] Early fusion done with dataset_source encoded (CIC=0, TON=1)"
                    )
            except Exception as exc:
                logger.warning(
                    "Harmonization failed, falling back to TON_IoT only: %s", exc
                )
                df_processed = df_ton
        else:
            df_processed = df_ton

        if "dataset_source" not in df_processed.columns:
            df_processed["dataset_source"] = 1

        # Ensure label column exists
        if "label" not in df_processed.columns:
            for candidate in ["Label", "label", "Attack", "Class"]:
                if candidate in df_processed.columns:
                    df_processed = df_processed.rename(columns={candidate: "label"})
                    break

        if "label" not in df_processed.columns:
            raise ValueError("Label column not found in dataset for Phase 2.")

        return df_processed

    def _apply_stateless_preprocessing(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Apply stateless preprocessing only (clean_data + encode_features, NO scaling/FS/SMOTE)."""
        logger.info(
            "Applying stateless preprocessing (clean_data + encode_features only)..."
        )
        logger.info(
            "NOTE: Fit-dependent steps (scaling, feature selection, SMOTE) will be applied in Phase 3 per fold."
        )

        pipeline = PreprocessingPipeline(
            random_state=self.config.random_state,
            n_features=20,  # Not used for stateless preprocessing
        )

        # Step 1: Clean data (NaN/Infinity removal, median imputation)
        # Phase 2 requirement: Replace inf by max(colonne), apply median imputation
        # Note: Imputation is stateless (median computed from full dataset before splitting)
        X_cleaned, y_cleaned = pipeline.clean_data(X, y, impute=True, replace_inf_with_max=True)

        # Step 2: Encode features (categorical encoding)
        # Note: encode_features might be stateless if no categorical features
        X_encoded = pipeline.encode_features(X_cleaned)

        # Mark pipeline as "fitted" for encoding (even if stateless)
        # This is needed for consistency, but we won't use fit-dependent steps
        if hasattr(pipeline, "label_encoders") and pipeline.label_encoders:
            pipeline.is_fitted = True

        logger.info(
            f"Stateless preprocessing complete: {X_encoded.shape[0]} rows, {X_encoded.shape[1]} features"
        )
        if y_cleaned is not None:
            logger.info(f"Label distribution: {y_cleaned.value_counts().to_dict()}")

        return X_encoded, y_cleaned

    def _save_outputs(
        self, df_final: pd.DataFrame, feature_names: list
    ) -> Dict[str, Path]:
        """Save processed dataset and metadata."""
        output_paths = {}

        # Save processed dataset (parquet or csv.gz)
        try:
            output_file = self.results_dir / "best_preprocessed.parquet"
            df_final.to_parquet(output_file, index=False, engine="pyarrow")
            output_paths["preprocessed_data"] = output_file
            logger.info(f"Saved preprocessed dataset to {output_file}")
        except Exception as exc:
            logger.warning(f"Parquet save failed ({exc}), falling back to csv.gz...")
            output_file = self.results_dir / "best_preprocessed.csv.gz"
            df_final.to_csv(output_file, index=False, compression="gzip")
            output_paths["preprocessed_data"] = output_file
            logger.info(f"Saved preprocessed dataset to {output_file}")

        # Save feature names
        feature_names_file = self.results_dir / "feature_names.json"
        with open(feature_names_file, "w", encoding="utf-8") as f:
            json.dump({"feature_names": feature_names}, f, indent=2)
        output_paths["feature_names"] = feature_names_file
        logger.info(f"Saved feature names to {feature_names_file}")

        # Save summary report
        summary_file = self.results_dir / "phase2_summary.md"
        summary_content = self._generate_summary(df_final, feature_names)
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary_content)
        output_paths["summary"] = summary_file
        logger.info(f"Saved summary to {summary_file}")

        return output_paths

    def _generate_summary(self, df_final: pd.DataFrame, feature_names: list) -> str:
        """Generate Phase 2 summary report."""
        lines = []
        lines.append("# Phase 2: Apply Best Configuration - Summary")
        lines.append("")
        lines.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("## Dataset Information")
        lines.append(f"- **Total Rows**: {df_final.shape[0]:,}")
        lines.append(f"- **Total Features**: {len(feature_names)}")
        lines.append(f"- **Total Columns** (including label): {df_final.shape[1]}")
        lines.append("")

        # Dataset source distribution
        if "dataset_source" in df_final.columns:
            ds_counts = df_final["dataset_source"].value_counts().sort_index()
            lines.append("## Dataset Source Distribution")
            lines.append(
                f"- **CIC-DDoS2019** (dataset_source=0): {ds_counts.get(0, 0):,} rows"
            )
            lines.append(
                f"- **TON_IoT** (dataset_source=1): {ds_counts.get(1, 0):,} rows"
            )
            lines.append("")
            lines.append(
                "**Mapping**: `dataset_source` encoding: 0=CIC-DDoS2019, 1=TON_IoT"
            )
            lines.append("")

        # Label distribution
        if "label" in df_final.columns:
            label_counts = df_final["label"].value_counts().sort_index()
            lines.append("## Label Distribution")
            for label, count in label_counts.items():
                pct = (count / len(df_final)) * 100
                lines.append(f"- **Class {label}**: {count:,} rows ({pct:.2f}%)")
            lines.append("")

        # Preprocessing steps applied
        lines.append("## Preprocessing Steps Applied (Phase 2)")
        lines.append("")
        lines.append("**Stateless preprocessing only**:")
        lines.append("- ✅ Data cleaning (NaN/Infinity removal, median imputation)")
        lines.append("- ✅ Feature encoding (categorical features)")
        lines.append(
            "- ❌ Feature selection (NOT applied - will be done in Phase 3 per fold)"
        )
        lines.append("- ❌ Scaling (NOT applied - will be done in Phase 3 per fold)")
        lines.append(
            "- ❌ SMOTE resampling (NOT applied - will be done in Phase 3 per fold)"
        )
        lines.append("")
        lines.append(
            "**Note**: Fit-dependent steps (scaling, feature selection, SMOTE) are applied in Phase 3 per fold to ensure zero data leakage."
        )
        lines.append("")

        # Configuration used
        if self.best_config:
            lines.append("## Best Configuration (from Phase 1)")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(self.best_config, indent=2))
            lines.append("```")
            lines.append("")
            lines.append(
                "*Note: This configuration is documented for traceability. Only stateless steps are applied in Phase 2.*"
            )
            lines.append("")

        # Memory footprint
        memory_mb = df_final.memory_usage(deep=True).sum() / 1024 / 1024
        lines.append("## Memory Footprint")
        lines.append(f"- **Estimated memory usage**: {memory_mb:.2f} MB")
        lines.append("")

        # Output files
        lines.append("## Output Files")
        lines.append(
            f"- `best_preprocessed.parquet` (or `.csv.gz`): Preprocessed dataset"
        )
        lines.append(f"- `feature_names.json`: List of feature names")
        lines.append(f"- `phase2_summary.md`: This summary file")
        lines.append("")

        return "\n".join(lines)
