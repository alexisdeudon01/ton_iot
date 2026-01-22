from abc import ABC, abstractmethod
from typing import Any, Optional
from src.core.contracts.artifacts import TableArtifact, ModelArtifact, PreprocessArtifact, AlignmentArtifact, PredictionArtifact

class ArtifactPort(ABC):
    @abstractmethod
    def save_table(self, artifact: TableArtifact) -> None:
        pass

    @abstractmethod
    def load_table(self, artifact_id: str) -> TableArtifact:
        pass

    @abstractmethod
    def save_model(self, artifact: ModelArtifact) -> None:
        pass

    @abstractmethod
    def load_model(self, artifact_id: str) -> ModelArtifact:
        pass

    @abstractmethod
    def save_preprocess(self, artifact: PreprocessArtifact) -> None:
        pass

    @abstractmethod
    def load_preprocess(self, artifact_id: str) -> PreprocessArtifact:
        pass

    @abstractmethod
    def save_alignment(self, artifact: AlignmentArtifact) -> None:
        pass

    @abstractmethod
    def load_alignment(self, artifact_id: str) -> AlignmentArtifact:
        pass

    @abstractmethod
    def save_prediction(self, artifact: PredictionArtifact) -> None:
        pass

    @abstractmethod
    def load_prediction(self, artifact_id: str) -> PredictionArtifact:
        pass

    @abstractmethod
    def get_path(self, artifact_id: str) -> str:
        pass
