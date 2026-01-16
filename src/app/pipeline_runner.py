#!/usr/bin/env python3
"""
Pipeline Runner: orchestration du pipeline IRP 5 phases
Sans dépendance UI (optionnelle via config.interactive)
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from ..config import PipelineConfig
from ..phases.phase1_config_search import Phase1ConfigSearch
# Imports conditionnels pour phases 2-5 (à créer)
try:
    from ..phases.phase2_apply_best_config import Phase2ApplyBestConfig
except ImportError:
    Phase2ApplyBestConfig = None

try:
    from ..phases.phase3_evaluation import Phase3Evaluation
except ImportError:
    Phase3Evaluation = None

try:
    from ..phases.phase4_ahp_preferences import Phase4AHPPreferences
except ImportError:
    Phase4AHPPreferences = None

try:
    from ..phases.phase5_topsis_ranking import Phase5TOPSISRanking
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
        self.phase_results = {}
        
        logger.info(f"PipelineRunner initialized (output_dir: {self.results_dir})")
    
    def run(self, phases: Optional[list] = None):
        """
        Run pipeline phases
        
        Args:
            phases: List of phase numbers to run (1-5). If None, runs all enabled phases.
        """
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
            phase1 = Phase1ConfigSearch(self.config)
            result1 = phase1.run()
            self.phase_results[1] = result1
            
            # Store best config for Phase 2
            self.best_config = result1.get('best_config')
        
        # Phase 2: Apply Best Config
        if 2 in phases:
            if Phase2ApplyBestConfig is None:
                logger.warning("Phase 2 not available (module not found)")
            else:
                if not hasattr(self, 'best_config'):
                    # Try to load from Phase 1 results
                    import json
                    best_config_file = self.results_dir / 'phase1_config_search' / 'best_config.json'
                    if best_config_file.exists():
                        with open(best_config_file) as f:
                            data = json.load(f)
                            self.best_config = data['config']
                    else:
                        raise ValueError("Phase 2 requires Phase 1 to run first or best_config.json")
                
                logger.info("\n" + "=" * 70)
                phase2 = Phase2ApplyBestConfig(self.config, self.best_config)
                result2 = phase2.run()
                self.phase_results[2] = result2
        
        # Phase 3: Evaluation
        if 3 in phases:
            if Phase3Evaluation is None:
                logger.warning("Phase 3 not available (module not found)")
            else:
                logger.info("\n" + "=" * 70)
                phase3 = Phase3Evaluation(self.config)
                result3 = phase3.run()
                self.phase_results[3] = result3
        
        # Phase 4: AHP Preferences
        if 4 in phases:
            if Phase4AHPPreferences is None:
                logger.warning("Phase 4 not available (module not found)")
            else:
                logger.info("\n" + "=" * 70)
                phase4 = Phase4AHPPreferences(self.config)
                result4 = phase4.run()
                self.phase_results[4] = result4
        
        # Phase 5: TOPSIS Ranking
        if 5 in phases:
            if Phase5TOPSISRanking is None:
                logger.warning("Phase 5 not available (module not found)")
            else:
                logger.info("\n" + "=" * 70)
                phase5 = Phase5TOPSISRanking(self.config)
                result5 = phase5.run()
                self.phase_results[5] = result5
        
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Results saved to: {self.results_dir}")
        
        return self.phase_results
