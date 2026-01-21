#!/usr/bin/env python3
"""
Pipeline Runner: orchestration du pipeline IRP 5 phases
Sans dépendance UI (optionnelle via config.interactive)
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import sys


# Import from config module (src/config.py)
_parent_dir = Path(__file__).parent.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from src.config import PipelineConfig
from src.phases.phase1_config_search import Phase1ConfigSearch
# Imports conditionnels pour phases 2-5 (à créer)
try:
    from src.phases.phase2_apply_best_config import Phase2ApplyBestConfig
except ImportError:
    Phase2ApplyBestConfig = None

try:
    from src.phases.phase3_evaluation import Phase3Evaluation
except ImportError:
    Phase3Evaluation = None

try:
    from src.phases.phase4_ahp_preferences import Phase4AHPPreferences
except ImportError:
    Phase4AHPPreferences = None

try:
    from src.phases.phase5_topsis_ranking import Phase5TOPSISRanking
except ImportError:
    Phase5TOPSISRanking = None

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrateur du pipeline IRP (5 phases)"""

    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline runner

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.results_dir = Path(config.output_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Results from each phase
        self.phase_results: Dict[int, Any] = {}
        self.best_config: Optional[Dict[str, Any]] = None

        logger.info(f"PipelineRunner initialized (output_dir: {self.results_dir})")

    def run(self, phases: Optional[list] = None):
        """
        Run pipeline phases sequentially.

        Each phase is an independent step in the IRP Research Pipeline:
        - Phase 1: Search for the best preprocessing configuration among 108 options.
        - Phase 2: Apply the best configuration (stateless steps) to the full dataset.
        - Phase 3: Comprehensive evaluation using cross-validation and multiple models.
        - Phase 4: Define AHP preferences for multi-criteria decision making.
        - Phase 5: Rank models using TOPSIS based on Phase 3 results and Phase 4 preferences.

        Args:
            phases: List of phase numbers to run (1-5). If None, runs all enabled phases.
        """
        logger.info("[VERBOSE] Starting PipelineRunner.run()")
        logger.info(f"[VERBOSE] Input Configuration: {self.config}")

        if phases is None:
            phases = []
            if self.config.phase1_search_enabled:
                phases.append(1)
            if self.config.phase2_enabled:
                phases.append(2)
            if self.config.phase3_enabled:
                phases.append(3)
            if self.config.phase4_enabled:
                phases.append(4)
            if self.config.phase5_enabled:
                phases.append(5)

        logger.info("=" * 70)
        logger.info("IRP RESEARCH PIPELINE - 5 PHASES")
        logger.info("=" * 70)
        logger.info(f"Running phases: {phases}")

        # Phase 1: Config Search
        if 1 in phases:
            logger.info("\n" + "=" * 70)
            logger.info("[PHASE 1] Preprocessing Configuration Search")
            logger.info("[DESCRIPTION] Evaluates 108 combinations of scaling, selection, and resampling.")
            logger.info("[INPUT] Raw datasets (TON_IoT, CIC-DDoS2019)")

            phase1 = Phase1ConfigSearch(self.config)
            result1 = phase1.run()
            self.phase_results[1] = result1

            # Store best config for Phase 2
            self.best_config = result1.get('best_config')
            logger.info(f"[OUTPUT] Phase 1 complete. Best config found: {self.best_config}")

        # Phase 2: Apply Best Config
        if 2 in phases:
            logger.info("\n" + "=" * 70)
            logger.info("[PHASE 2] Apply Best Configuration")
            logger.info("[DESCRIPTION] Applies stateless preprocessing steps to the full dataset.")

            if Phase2ApplyBestConfig is None:
                logger.warning("Phase 2 not available (module not found)")
            else:
                if self.best_config is None:
                    # Try to load from Phase 1 results
                    import json
                    best_config_file = self.results_dir / 'phase1_config_search' / 'best_config.json'
                    if best_config_file.exists():
                        with open(best_config_file) as f:
                            data = json.load(f)
                            self.best_config = data['config']
                    else:
                        raise ValueError("Phase 2 requires Phase 1 to run first or best_config.json")

                if self.best_config is not None:
                    logger.info(f"[INPUT] Best configuration: {self.best_config}")
                    phase2 = Phase2ApplyBestConfig(self.config, self.best_config)
                    result2 = phase2.run()
                    self.phase_results[2] = result2
                    logger.info(f"[OUTPUT] Phase 2 complete. Processed data saved to {result2.get('output_paths', {}).get('preprocessed_data')}")
                else:
                    raise ValueError("best_config is still None after attempt to load")

        # Phase 3: Evaluation
        if 3 in phases:
            logger.info("\n" + "=" * 70)
            logger.info("[PHASE 3] Comprehensive Evaluation")
            logger.info("[DESCRIPTION] Trains and evaluates multiple models (CNN, TabNet, etc.) using cross-validation.")
            logger.info("[INPUT] Preprocessed dataset from Phase 2")

            if Phase3Evaluation is None:
                logger.warning("Phase 3 not available (module not found)")
            else:
                phase3 = Phase3Evaluation(self.config)
                result3 = phase3.run()
                self.phase_results[3] = result3
                logger.info(f"[OUTPUT] Phase 3 complete. Metrics and visualizations generated.")

        # Phase 4: AHP Preferences
        if 4 in phases:
            logger.info("\n" + "=" * 70)
            logger.info("[PHASE 4] AHP Preferences")
            logger.info("[DESCRIPTION] Defines user preferences for model ranking using Analytic Hierarchy Process.")

            if Phase4AHPPreferences is None:
                logger.warning("Phase 4 not available (module not found)")
            else:
                phase4 = Phase4AHPPreferences(self.config)
                result4 = phase4.run()
                self.phase_results[4] = result4
                logger.info(f"[OUTPUT] Phase 4 complete. Preference weights calculated.")

        # Phase 5: TOPSIS Ranking
        if 5 in phases:
            logger.info("\n" + "=" * 70)
            logger.info("[PHASE 5] TOPSIS Ranking")
            logger.info("[DESCRIPTION] Ranks models using TOPSIS based on Phase 3 metrics and Phase 4 weights.")
            logger.info("[INPUT] Phase 3 metrics and Phase 4 weights")

            if Phase5TOPSISRanking is None:
                logger.warning("Phase 5 not available (module not found)")
            else:
                phase5 = Phase5TOPSISRanking(self.config)
                result5 = phase5.run()
                self.phase_results[5] = result5
                logger.info(f"[OUTPUT] Phase 5 complete. Final model rankings generated.")

        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Results saved to: {self.results_dir}")

        return self.phase_results
