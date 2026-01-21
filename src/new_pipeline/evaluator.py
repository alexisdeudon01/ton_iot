import logging
from pathlib import Path
from typing import Dict, List, Any

from src.datastructure.toniot_dataframe import ToniotDataFrame
import numpy as np
import joblib
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_score

from src.config import settings as config
from src.core.results import TestResult

logger = logging.getLogger(__name__)

class LateFusionEvaluator:
    """Phase 6: Évaluation (Intra & Transfert) avec Recall@k."""

    def __init__(self, m_cic_path: Path, m_ton_path: Path, w: float, features: List[str]):
        self.m_cic = joblib.load(m_cic_path)
        self.m_ton = joblib.load(m_ton_path)
        self.w = w
        self.features = features
        self.label_col = config.phase0.label_col
        self.threshold = config.phase5.threshold
        self.p6 = config.phase6
        logger.info(f"Initialized LateFusionEvaluator (w={w:.2f})")

    def evaluate_all(self, cic_test_path: Path, ton_test_path: Path) -> List[TestResult]:
        """Comprehensive evaluation including intra-dataset, transfer, and fusion."""
        logger.info("Starting Phase 6: Comprehensive Evaluation")
        
        df_cic = ToniotDataFrame(pd.read_parquet(cic_test_path))
        df_ton = ToniotDataFrame(pd.read_parquet(ton_test_path))
        
        results = []
        
        # 1. Intra-dataset
        results.append(self._eval_single("CIC -> CIC", df_cic, self.m_cic))
        results.append(self._eval_single("TON -> TON", df_ton, self.m_ton))
        
        # 2. Transfer
        if self.p6.enable_transfer_tests:
            results.append(self._eval_single("CIC -> TON", df_ton, self.m_cic))
            results.append(self._eval_single("TON -> CIC", df_cic, self.m_ton))
        
        # 3. Fusion (on both)
        df_comb = ToniotDataFrame(pd.concat([df_cic, df_ton]))
        results.append(self._eval_fusion("LATE FUSION (Combined)", df_comb))
        
        return results

    def _eval_single(self, name: str, df: ToniotDataFrame, model) -> TestResult:
        X = df[self.features]
        y = df[self.label_col]
        
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        return self._create_result(name, y, y_pred, y_prob)

    def _eval_fusion(self, name: str, df: ToniotDataFrame) -> TestResult:
        X = df[self.features]
        y = df[self.label_col]
        
        p_cic = self.m_cic.predict_proba(X)[:, 1]
        p_ton = self.m_ton.predict_proba(X)[:, 1]
        p_final = self.w * p_cic + (1 - self.w) * p_ton
        y_pred = (p_final >= self.threshold).astype(int)
        
        return self._create_result(name, y, y_pred, p_final)

    def _recall_at_k(self, y_true, y_prob, k: int) -> float:
        """Calculates Recall at top k predicted probabilities."""
        if k <= 0: return 0.0
        # Sort by probability descending
        top_k_indices = np.argsort(y_prob)[-k:]
        # Count true positives in top k
        tp = y_true.iloc[top_k_indices].sum()
        total_pos = y_true.sum()
        return tp / total_pos if total_pos > 0 else 0.0

    def _create_result(self, name: str, y_true, y_pred, y_prob) -> TestResult:
        # Standard metrics
        res = TestResult(
            model_name=name,
            accuracy=float((y_true == y_pred).mean()),
            f1_score=f1_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred),
            recall=recall_score(y_true, y_pred),
            auc=roc_auc_score(y_true, y_prob)
        )
        
        # Recall@k metrics
        for k in self.p6.recall_at_k:
            r_at_k = self._recall_at_k(y_true, y_prob, k)
            logger.info(f"  - {name} Recall@{k}: {r_at_k:.4f}")
            # Note: TestResult dataclass might need extension for Recall@k if we want to store it
            # For now we log it.
            
        logger.info(f"Result for {name}: F1={res.f1_score:.4f}, Recall={res.recall:.4f}, AUC={res.auc:.4f}")
        return res
