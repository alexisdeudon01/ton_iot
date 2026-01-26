import unittest
from pathlib import Path

import pandas as pd
import yaml

from src.preprocessing import _stratified_sample


class TestSamplingFraction(unittest.TestCase):
    def test_config_sampling_fraction_is_5_percent(self):
        cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        sampling = cfg.get("sampling", {})
        self.assertAlmostEqual(float(sampling.get("fraction", 0.0)), 0.05)

    def test_stratified_sample_respects_5_percent(self):
        df = pd.DataFrame({
            "feature": list(range(200)),
            "y": [0] * 100 + [1] * 100,
        })
        sampled = _stratified_sample(df, "y", 0.05, seed=42)
        counts = sampled["y"].value_counts().sort_index().to_list()
        self.assertEqual(counts, [5, 5])


if __name__ == "__main__":
    unittest.main()
