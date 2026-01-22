import os
import json
from typing import Optional
from src.core.ports.artifacts import ArtifactPort
from src.core.contracts.artifacts import TableArtifact, ModelArtifact, PreprocessArtifact, AlignmentArtifact, PredictionArtifact

class FileArtifactStore(ArtifactPort):
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _get_meta_path(self, artifact_id: str) -> str:
        return os.path.join(self.base_dir, f"{artifact_id}.json")

    def save_table(self, artifact: TableArtifact) -> None:
        with open(self._get_meta_path(artifact.artifact_id), "w") as f:
            f.write(artifact.model_dump_json())

    def load_table(self, artifact_id: str) -> TableArtifact:
        with open(self._get_meta_path(artifact_id), "r") as f:
            return TableArtifact.model_validate_json(f.read())

    def save_model(self, artifact: ModelArtifact) -> None:
        with open(self._get_meta_path(artifact.artifact_id), "w") as f:
            f.write(artifact.model_dump_json())

    def load_model(self, artifact_id: str) -> ModelArtifact:
        with open(self._get_meta_path(artifact_id), "r") as f:
            return ModelArtifact.model_validate_json(f.read())

    def save_preprocess(self, artifact: PreprocessArtifact) -> None:
        with open(self._get_meta_path(artifact.artifact_id), "w") as f:
            f.write(artifact.model_dump_json())

    def load_preprocess(self, artifact_id: str) -> PreprocessArtifact:
        with open(self._get_meta_path(artifact_id), "r") as f:
            return PreprocessArtifact.model_validate_json(f.read())

    def save_alignment(self, artifact: AlignmentArtifact) -> None:
        with open(self._get_meta_path(artifact.artifact_id), "w") as f:
            f.write(artifact.model_dump_json())

    def load_alignment(self, artifact_id: str) -> AlignmentArtifact:
        with open(self._get_meta_path(artifact_id), "r") as f:
            return AlignmentArtifact.model_validate_json(f.read())

    def save_prediction(self, artifact: PredictionArtifact) -> None:
        with open(self._get_meta_path(artifact.artifact_id), "w") as f:
            f.write(artifact.model_dump_json())

    def load_prediction(self, artifact_id: str) -> PredictionArtifact:
        with open(self._get_meta_path(artifact_id), "r") as f:
            return PredictionArtifact.model_validate_json(f.read())

    def get_path(self, artifact_id: str) -> str:
        # This is a simplified version, in reality we might store the actual data path in the meta
        # For now, we assume the data file is in the same directory with a specific extension
        # But the artifacts themselves already contain the 'path' field.
        try:
            with open(self._get_meta_path(artifact_id), "r") as f:
                data = json.load(f)
                return data.get("path") or data.get("model_path") or data.get("preprocess_path")
        except:
            return os.path.join(self.base_dir, artifact_id)
