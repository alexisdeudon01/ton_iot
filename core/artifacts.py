import os
import joblib
import polars as pl
from typing import Any, Optional

class ArtifactManager:
    """
    Gère la persistance des données (Parquet) et des objets Python (Joblib).
    Centralise l'accès aux fichiers pour assurer la traçabilité.
    """
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save_table(self, df: pl.DataFrame, name: str) -> str:
        """Sauvegarde un DataFrame Polars au format Parquet avec compression zstd."""
        path = os.path.join(self.base_dir, f"{name}.parquet")
        df.write_parquet(path, compression="zstd")
        return path

    def load_table(self, name: str) -> pl.DataFrame:
        """Charge un fichier Parquet."""
        path = os.path.join(self.base_dir, f"{name}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifact {name} non trouvé à {path}")
        return pl.read_parquet(path)

    def save_model(self, model: Any, name: str) -> str:
        """Sauvegarde un modèle ou un objet Python via Joblib."""
        path = os.path.join(self.base_dir, f"{name}.joblib")
        joblib.dump(model, path)
        return path

    def load_model(self, name: str) -> Any:
        """Charge un objet Joblib."""
        path = os.path.join(self.base_dir, f"{name}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modèle {name} non trouvé à {path}")
        return joblib.load(path)

    def exists(self, name: str, ext: str = "parquet") -> bool:
        """Vérifie l'existence d'un artefact."""
        return os.path.exists(os.path.join(self.base_dir, f"{name}.{ext}"))

    def get_path(self, name: str, ext: str = "parquet") -> str:
        """Retourne le chemin complet d'un artefact."""
        return os.path.join(self.base_dir, f"{name}.{ext}")
