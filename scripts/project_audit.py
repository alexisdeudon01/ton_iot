from __future__ import annotations

from pathlib import Path
from typing import Dict, List


REQUIRED_DIRS = [
    Path("data/raw"),
    Path("data/sample"),
    Path("data/processed"),
    Path("src/preprocessing"),
    Path("src/models"),
    Path("src/mcdm"),
    Path("src/visualization"),
    Path("graphs"),
    Path("results"),
]

ARTIFACT_DIRS = [
    Path("Outputs"),
    Path("output"),
    Path("output_test"),
    Path("graph"),
    Path("work"),
    Path(".pytest_cache"),
    Path("__pycache__"),
    Path(".cursor"),
    Path(".venv_results"),
    Path("rr"),
    Path("other"),
    Path("spec"),
    Path("reports"),
]

ARTIFACT_FILES = [
    Path("legacy_main_gui.py"),
    Path("expert_pipeline.py"),
    Path("main_test.py"),
    Path("implementation_plan.md"),
    Path("generate_results_chapter.py"),
    Path("generate_universal_feature_distributions.py"),
    Path("universal_features.py"),
    Path("verify_irp_compliance.py"),
    Path("structureword.json"),
    Path("test_toniot.code-workspace"),
    Path("toniot.code-workspace"),
    Path("req.py"),
]


def ensure_structure() -> List[Path]:
    created: List[Path] = []
    for path in REQUIRED_DIRS:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
    return created


def purge_log_files(root: Path) -> List[Path]:
    removed: List[Path] = []
    for path in root.rglob("*.log"):
        try:
            path.unlink()
            removed.append(path)
        except OSError:
            continue
    for path in root.rglob("*.log.*"):
        try:
            path.unlink()
            removed.append(path)
        except OSError:
            continue
    return removed


def collect_cleanup_candidates(root: Path) -> Dict[str, List[Path]]:
    candidates = {"dirs": [], "files": []}

    for path in ARTIFACT_DIRS:
        full = root / path
        if full.exists():
            candidates["dirs"].append(full)

    for path in ARTIFACT_FILES:
        full = root / path
        if full.exists():
            candidates["files"].append(full)

    return candidates


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    created = ensure_structure()
    removed_logs = purge_log_files(project_root)
    candidates = collect_cleanup_candidates(project_root)

    print(f"[audit] created_dirs: {[str(p) for p in created]}")
    print(f"[audit] removed_logs: {len(removed_logs)}")
    print(f"[audit] candidate_dirs: {[str(p) for p in candidates['dirs']]}")
    print(f"[audit] candidate_files: {[str(p) for p in candidates['files']]}")
