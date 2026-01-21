import os
import json
from typing import List, Any
from src.core.ports.interfaces import IArtifactStore
from pydantic import BaseModel

class FileArtifactStoreAdapter(IArtifactStore):
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save_artifact(self, name: str, artifact: Any):
        path = os.path.join(self.base_dir, f"{name}.json")
        if isinstance(artifact, BaseModel):
            with open(path, "w") as f:
                f.write(artifact.model_dump_json(indent=2))
        else:
            # Fallback for non-pydantic objects if needed
            with open(path, "w") as f:
                json.dump(artifact, f, indent=2)

    def load_artifact(self, name: str) -> Any:
        path = os.path.join(self.base_dir, f"{name}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)

    def list_artifacts(self) -> List[str]:
        return [f.replace(".json", "") for f in os.listdir(self.base_dir) if f.endswith(".json")]
