"""
Point d'entrée principal - Framework d'évaluation DDoS pour PME
Usage: python main.py [--config config.yaml] [--skip-training] [--only-viz]

Auteur: Alexis Deudon , cours Independt research porjct, reference module: COM00151M
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict



def _in_venv() -> bool:
    return sys.prefix != sys.base_prefix or bool(os.environ.get("VIRTUAL_ENV"))


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _requirements_hash(req_path: Path) -> str:
    content = req_path.read_text(encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _install_requirements(python_exec: str, req_path: Path, venv_root: Path) -> None:
    if not req_path.exists():
        return

    subprocess.check_call([python_exec, "-m", "ensurepip", "--upgrade"])

    hash_path = venv_root / ".requirements.sha256"
    req_hash = _requirements_hash(req_path)
    needs_install = True

    if hash_path.exists() and hash_path.read_text(encoding="utf-8").strip() == req_hash:
        try:
            subprocess.check_call([python_exec, "-m", "pip", "check"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            needs_install = False
        except subprocess.CalledProcessError:
            needs_install = True

    if needs_install:
        subprocess.check_call([python_exec, "-m", "pip", "install", "-r", str(req_path)])
        hash_path.write_text(req_hash, encoding="utf-8")


def _ensure_venv_and_deps() -> None:
    if os.environ.get("DDOS_SKIP_BOOTSTRAP") == "1":
        return

    root_dir = Path(__file__).resolve().parent
    venv_dir = root_dir / ".venv"
    req_path = root_dir / "requirements.txt"

    if not _in_venv() or Path(sys.prefix).resolve() != venv_dir.resolve():
        if not venv_dir.exists():
            subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        venv_py = _venv_python(venv_dir)
        _install_requirements(str(venv_py), req_path, venv_dir)
        os.execv(str(venv_py), [str(venv_py)] + sys.argv)

    _install_requirements(sys.executable, req_path, Path(sys.prefix))


def _set_random_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def load_config(path: str | Path) -> Dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        import yaml

        return yaml.safe_load(fh)


def _remove_venv_if_docker_ready(docker_status: Dict[str, object], root_dir: Path) -> None:
    venv_dir = root_dir / ".venv"
    if not docker_status.get("docker_ready"):
        return
    if _in_venv():
        return
    if venv_dir.exists():
        import shutil

        shutil.rmtree(venv_dir, ignore_errors=True)


def _preflight_checks() -> Dict[str, object]:
    if os.environ.get("DDOS_PREFLIGHT_DONE") == "1":
        return {}

    os.environ["DDOS_PREFLIGHT_DONE"] = "1"
    root_dir = Path(__file__).resolve().parent

    from scripts.check_docker import check_docker_environment
    from scripts.project_audit import collect_cleanup_candidates, ensure_structure, purge_log_files

    docker_status = check_docker_environment(print_report=True)
    removed_logs = purge_log_files(root_dir)
    created_dirs = ensure_structure()
    candidates = collect_cleanup_candidates(root_dir)
    if created_dirs:
        print(f"[audit] Dossiers créés: {[str(p) for p in created_dirs]}")
    if removed_logs:
        print(f"[audit] Logs supprimés: {len(removed_logs)}")
    if candidates.get("dirs") or candidates.get("files"):
        print(f"[audit] Candidats suppression (dirs): {[str(p) for p in candidates.get('dirs', [])]}")
        print(f"[audit] Candidats suppression (files): {[str(p) for p in candidates.get('files', [])]}")
    _remove_venv_if_docker_ready(docker_status, root_dir)
    return docker_status


def _ensure_datasets_present(config: Dict) -> None:
    datasets_cfg = config.get("datasets", {})
    cic_path = datasets_cfg.get("cic_ddos2019") or datasets_cfg.get("cic_path")
    ton_path = datasets_cfg.get("ton_iot") or datasets_cfg.get("ton_path")

    missing = []
    if not cic_path or not Path(cic_path).exists():
        missing.append(f"CIC-DDoS2019 -> {cic_path}")
    if not ton_path or not Path(ton_path).exists():
        missing.append(f"TON_IoT -> {ton_path}")

    if missing:
        print("[data] Dataset introuvable.")
        for item in missing:
            print(f"[data] {item}")
        print("[data] Le dataset est disponible sur Google Drive.")
        sys.exit(1)


def main() -> None:
    _preflight_checks()
    _ensure_venv_and_deps()

    parser = argparse.ArgumentParser(description="Framework d'évaluation DDoS pour PME")
    parser.add_argument("--config", default="config.yaml", help="Chemin du fichier config.yaml")
    parser.add_argument("--skip-training", action="store_true", help="Ignore l'entraînement")
    parser.add_argument("--only-viz", action="store_true", help="Génère uniquement les figures")
    args = parser.parse_args()

    config = load_config(args.config)
    _set_random_seed(int(config.get("validation", {}).get("random_state", 42)))
    from src.preprocessing import load_and_preprocess
    from src.models import train_and_evaluate_all
    from src.mcdm import run_ahp_topsis, run_sensitivity_analysis
    from src.visualization import generate_all_figures

    def log(message: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {message}")

    if args.only_viz:
        log("Generating figures only")
        generate_all_figures()
        return

    log("Loading and preprocessing datasets")
    load_and_preprocess(config)

    if not args.skip_training:
        log("Training and evaluating models")
        train_and_evaluate_all(config)

    log("Running AHP-TOPSIS ranking")
    run_ahp_topsis(config)
    log("Running sensitivity analysis")
    run_sensitivity_analysis(config)
    log("Generating figures")
    generate_all_figures()

    # Save summary JSON
    results_dir = Path("results")
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics_csv": str(results_dir / "metrics.csv"),
        "decision_matrix_csv": str(results_dir / "decision_matrix.csv"),
        "topsis_ranking_csv": str(results_dir / "topsis_ranking.csv"),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
