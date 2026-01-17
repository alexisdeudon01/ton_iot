import pandas as pd

from src.config import PipelineConfig
from src.phases.phase2_apply_best_config import Phase2ApplyBestConfig


def test_phase2_outputs(tmp_path, monkeypatch):
    config = PipelineConfig(output_dir=str(tmp_path), test_mode=True, sample_ratio=0.01)
    phase2 = Phase2ApplyBestConfig(config, best_config={"dummy": True})

    def fake_load_and_harmonize():
        data = {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [10.0, 20.0, 30.0],
            "dataset_source": [0, 1, 1],
            "label": [0, 1, 0],
        }
        return pd.DataFrame(data)

    monkeypatch.setattr(phase2, "_load_and_harmonize_datasets", fake_load_and_harmonize)

    result = phase2.run()

    output_paths = result["output_paths"]
    assert output_paths["preprocessed_data"].exists()
    assert output_paths["feature_names"].exists()
    assert output_paths["summary"].exists()
