import os
import json
import time
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.app.pipeline.registry import TaskRegistry
from src.mcdm.metrics_utils import compute_f_perf
from mcdm.topsis_tppis_report import build_df_sources_from_run_report, normalize_matrix


@TaskRegistry.register("T20_WeightSampleVariation")
class T20_WeightSampleVariation(Task):
    """Vary user AHP weights and sampling size, then plot sensitivity."""

    def _load_user_weights(self) -> np.ndarray:
        weights = np.array([0.70, 0.15, 0.15], dtype=float)
        try:
            import yaml
            with open("config/pipeline.yaml", "r") as f:
                cfg = yaml.safe_load(f)
        except Exception:
            cfg = None
        if isinstance(cfg, dict):
            pillars = cfg.get("mcdm_hierarchy", {}).get("pillars", [])
            if len(pillars) >= 3:
                weights = np.array([
                    float(pillars[0].get("weight", weights[0])),
                    float(pillars[1].get("weight", weights[1])),
                    float(pillars[2].get("weight", weights[2])),
                ])
        if weights.sum() <= 0:
            weights = np.array([0.70, 0.15, 0.15], dtype=float)
        return weights / weights.sum()

    def _topsis_cc(self, norm_matrix: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
        M = norm_matrix.to_numpy(dtype=float)
        w = weights / weights.sum()
        weighted = M * w
        ideal = np.array([weighted[:, 0].max(), weighted[:, 1].max(), weighted[:, 2].min()])
        anti = np.array([weighted[:, 0].min(), weighted[:, 1].min(), weighted[:, 2].max()])
        d_pos = np.linalg.norm(weighted - ideal, axis=1)
        d_neg = np.linalg.norm(weighted - anti, axis=1)
        return d_neg / (d_pos + d_neg + 1e-9)

    def _vary_weight(self, norm_matrix: pd.DataFrame, base_weights: np.ndarray, idx: int, values: list) -> pd.DataFrame:
        models = norm_matrix.index.tolist()
        records = []
        for w in values:
            remaining = max(0.0, 1.0 - w)
            other_idx = [i for i in range(3) if i != idx]
            base_other = base_weights[other_idx]
            if base_other.sum() <= 0:
                split = np.array([remaining / 2, remaining / 2])
            else:
                split = remaining * (base_other / base_other.sum())
            weights = base_weights.copy()
            weights[idx] = w
            weights[other_idx[0]] = split[0]
            weights[other_idx[1]] = split[1]
            cc = self._topsis_cc(norm_matrix, weights)
            for model, score in zip(models, cc):
                records.append({"model": model, "varied_weight": w, "cc": score})
        return pd.DataFrame(records)

    def _plot_weight_variation(self, df: pd.DataFrame, title: str, output_path: str) -> None:
        plt.figure(figsize=(10, 6))
        for model in sorted(df["model"].unique()):
            sub = df[df["model"] == model]
            plt.plot(sub["varied_weight"] * 100, sub["cc"], marker="o", label=model)
        plt.xlabel("Weight (%)")
        plt.ylabel("TOPSIS closeness (CC)")
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _weight_grid_evaluation(self, norm_matrix: pd.DataFrame, output_dir: str) -> str:
        """Evaluate all weight combinations that sum to 1 (within tolerance)."""
        os.makedirs(output_dir, exist_ok=True)
        weight_ranges = np.round(np.arange(0.0, 1.01, 0.1), 2)
        models = norm_matrix.index.tolist()
        records = []

        for w_perf in weight_ranges:
            for w_expl in weight_ranges:
                for w_res in weight_ranges:
                    total = w_perf + w_expl + w_res
                    if abs(total - 1.0) > 0.01:
                        continue
                    weights = np.array([w_perf, w_expl, w_res], dtype=float)
                    cc = self._topsis_cc(norm_matrix, weights)
                    winner_idx = int(np.argmax(cc))
                    records.append({
                        "w_perf": w_perf,
                        "w_expl": w_expl,
                        "w_res": w_res,
                        "winner": models[winner_idx],
                        "winner_cc": float(cc[winner_idx]),
                    })

        df_grid = pd.DataFrame(records)
        grid_csv = os.path.join(output_dir, "weights_grid_results.csv")
        df_grid.to_csv(grid_csv, index=False)

        # Winner frequency bar chart
        counts = df_grid["winner"].value_counts().sort_index()
        plt.figure(figsize=(8, 5))
        plt.bar(counts.index, counts.values, color="#3498db")
        plt.title("Winner frequency across weight combinations")
        plt.xlabel("Algorithm")
        plt.ylabel("Count")
        plt.tight_layout()
        bar_path = os.path.join(output_dir, "weights_grid_winner_counts.png")
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Heatmap of winner by (w_perf, w_expl) with w_res = 1 - sum
        valid = df_grid.copy()
        valid["w_res_calc"] = (1.0 - valid["w_perf"] - valid["w_expl"]).round(2)
        valid = valid[valid["w_res_calc"] >= 0.0]
        winner_map = {name: idx for idx, name in enumerate(sorted(models))}
        valid["winner_idx"] = valid["winner"].map(winner_map)

        pivot = valid.pivot_table(
            index="w_perf",
            columns="w_expl",
            values="winner_idx",
            aggfunc="first",
        )
        plt.figure(figsize=(9, 7))
        cmap = plt.cm.get_cmap("tab10", len(winner_map))
        plt.imshow(pivot.values, origin="lower", cmap=cmap, aspect="auto")
        plt.xticks(range(len(pivot.columns)), [f"{v:.1f}" for v in pivot.columns])
        plt.yticks(range(len(pivot.index)), [f"{v:.1f}" for v in pivot.index])
        plt.xlabel("w_expl")
        plt.ylabel("w_perf")
        plt.title("Winner map (w_res = 1 - w_perf - w_expl)")

        handles = [
            plt.Line2D([0], [0], marker='s', linestyle='', color=cmap(i), label=name)
            for name, i in winner_map.items()
        ]
        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, "weights_grid_winner_map.png")
        plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        plt.close()

        return grid_csv
    def _sampling_variation(self, output_dir: str) -> str:
        pred_cic_path = os.path.join("work", "data", "predictions_cic.parquet")
        pred_ton_path = os.path.join("work", "data", "predictions_ton.parquet")
        if not (os.path.exists(pred_cic_path) and os.path.exists(pred_ton_path)):
            return ""
        df_cic = pl.read_parquet(pred_cic_path).filter(pl.col("split") == "test")
        df_ton = pl.read_parquet(pred_ton_path).filter(pl.col("split") == "test")
        if df_cic.is_empty() or df_ton.is_empty():
            return ""

        models = sorted(set(df_cic["model"].unique().to_list()) & set(df_ton["model"].unique().to_list()))
        if not models:
            return ""

        fractions = [i / 10 for i in range(1, 11)]
        results = {m: [] for m in models}
        from sklearn.metrics import f1_score, recall_score, roc_auc_score
        for frac in fractions:
            for model in models:
                df_cic_m = df_cic.filter(pl.col("model") == model)
                df_ton_m = df_ton.filter(pl.col("model") == model)
                if df_cic_m.is_empty() or df_ton_m.is_empty():
                    results[model].append(0.0)
                    continue
                n_cic = max(1, int(len(df_cic_m) * frac))
                n_ton = max(1, int(len(df_ton_m) * frac))
                sample_cic = df_cic_m.sample(n=n_cic, seed=int(42 + frac * 100)).to_pandas()
                sample_ton = df_ton_m.sample(n=n_ton, seed=int(84 + frac * 100)).to_pandas()
                y_true = np.concatenate([sample_cic["y_true"].to_numpy(), sample_ton["y_true"].to_numpy()])
                y_score = np.concatenate([sample_cic["proba"].to_numpy(), sample_ton["proba"].to_numpy()])
                y_pred = (y_score >= 0.5).astype(int)
                f1 = float(f1_score(y_true, y_pred, zero_division=0))
                recall = float(recall_score(y_true, y_pred, zero_division=0))
                try:
                    auc = float(roc_auc_score(y_true, y_score))
                except Exception:
                    auc = 0.5
                gap = 0.0
                results[model].append(float(compute_f_perf(f1, recall, auc, gap)))

        plt.figure(figsize=(10, 6))
        for model, values in results.items():
            plt.plot([int(f * 100) for f in fractions], values, marker="o", label=model)
        plt.xlabel("Sampling (%)")
        plt.ylabel("f_perf")
        plt.title("Sampling size variation (f_perf)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(output_dir, "sampling_size_variation.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return out_path

    def run(self, context: DAGContext) -> TaskResult:
        start_ts = time.time()
        report_path = os.path.join("reports", "run_report.json")
        if not os.path.exists(report_path):
            return TaskResult(task_name=self.name, status="failed", duration_s=time.time() - start_ts, error="run_report.json not found")

        with open(report_path, "r") as f:
            run_report = json.load(f)

        df_sources = build_df_sources_from_run_report(run_report)
        if df_sources.empty:
            return TaskResult(task_name=self.name, status="failed", duration_s=time.time() - start_ts, error="No fused models in run_report")

        base_weights = self._load_user_weights()
        context.logger.info("mcdm", "User weights loaded", weights=base_weights.tolist())

        norm_matrix = normalize_matrix(df_sources.set_index("model")[["f_perf", "f_expl", "f_res"]], ["f_perf", "f_expl", "f_res"])

        out_dir = os.path.join("graph", "decision", "variations", "weights")
        os.makedirs(out_dir, exist_ok=True)

        values = [i / 10 for i in range(1, 10)]
        perf_df = self._vary_weight(norm_matrix, base_weights, 0, values)
        expl_df = self._vary_weight(norm_matrix, base_weights, 1, values)
        res_df = self._vary_weight(norm_matrix, base_weights, 2, values)

        perf_path = os.path.join(out_dir, "weights_performance.png")
        expl_path = os.path.join(out_dir, "weights_explainability.png")
        res_path = os.path.join(out_dir, "weights_resources.png")
        self._plot_weight_variation(perf_df, "Weight variation: Performance", perf_path)
        self._plot_weight_variation(expl_df, "Weight variation: Explainability", expl_path)
        self._plot_weight_variation(res_df, "Weight variation: Resources", res_path)

        perf_df.to_csv(os.path.join(out_dir, "weights_performance.csv"), index=False)
        expl_df.to_csv(os.path.join(out_dir, "weights_explainability.csv"), index=False)
        res_df.to_csv(os.path.join(out_dir, "weights_resources.csv"), index=False)

        sample_out_dir = os.path.join("graph", "decision", "variations", "sampling_user")
        os.makedirs(sample_out_dir, exist_ok=True)
        sample_plot = self._sampling_variation(sample_out_dir)

        grid_out_dir = os.path.join("graph", "decision", "variations", "weights_grid")
        grid_csv = self._weight_grid_evaluation(norm_matrix, grid_out_dir)

        context.logger.info("mcdm", "Weight and sampling variation complete", outputs=[perf_path, expl_path, res_path, sample_plot, grid_csv])
        return TaskResult(
            task_name=self.name,
            status="ok",
            duration_s=time.time() - start_ts,
            outputs=[perf_path, expl_path, res_path, sample_plot, grid_csv],
        )
