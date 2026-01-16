#!/usr/bin/env python3
import os
import re
import sys
import shutil
import subprocess
from pathlib import Path

PROJECT_FILES = [
    "requirements.txt",
    "README.md",
    "data_training.py",
    "RL_training.py",
]

OPTIONAL_FILES = [
    "train_test_network.csv",
]

SUSPECT_IMPORTS = {
    "data_training.py": ["google.colab"],   # casse en local
    "RL_training.py": ["gymnasium"],        # si tu changes vers gymnasium un jour
}

def sh(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, out.strip()
    except subprocess.CalledProcessError as e:
        return e.returncode, e.output.strip()
    except Exception as e:
        return 1, str(e)

def ok(msg): print(f"[OK] {msg}")
def warn(msg): print(f"[WARN] {msg}")
def bad(msg): print(f"[FAIL] {msg}")

def file_contains(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False

def parse_requirements(req_path: Path):
    reqs = []
    if not req_path.exists():
        return reqs
    for line in req_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
    return reqs

def pip_show(pkg: str):
    code, out = sh([sys.executable, "-m", "pip", "show", pkg])
    if code != 0:
        return None
    m = re.search(r"^Version:\s*(.+)$", out, flags=re.MULTILINE)
    return m.group(1).strip() if m else "unknown"

def import_check(mod: str):
    code, out = sh([sys.executable, "-c", f"import {mod}; print(getattr({mod}, '__version__', 'ok'))"])
    return code == 0, out

def main():
    root = Path(os.getcwd())
    print("=== Project directory check ===")
    print("Root:", root)
    print("Python:", sys.version.split()[0])

    # 1) Repo sanity (git)
    if (root / ".git").exists():
        ok(".git found (looks like a git repo)")
        code, out = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        if code == 0:
            ok(f"git branch: {out}")
        else:
            warn("git exists but branch read failed")
    else:
        warn(".git not found (maybe not at repo root)")

    # 2) Required files
    print("\n=== Required files ===")
    missing = []
    for f in PROJECT_FILES:
        p = root / f
        if p.exists():
            ok(f"{f} present")
        else:
            bad(f"{f} missing")
            missing.append(f)

    print("\n=== Optional files ===")
    for f in OPTIONAL_FILES:
        p = root / f
        if p.exists():
            ok(f"{f} present")
        else:
            warn(f"{f} not found (might be ok depending on your workflow)")

    # 3) Virtual env detection
    print("\n=== Virtual environment ===")
    in_venv = (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix) or bool(os.environ.get("VIRTUAL_ENV"))
    if in_venv:
        ok(f"Running inside venv: {os.environ.get('VIRTUAL_ENV', sys.prefix)}")
    else:
        warn("Not inside a venv (recommended: python3 -m venv .venv && source .venv/bin/activate)")

    # 4) pip availability
    if shutil.which("pip") or shutil.which("pip3"):
        ok("pip is available")
    else:
        bad("pip not found in PATH")

    # 5) Read requirements + check install status
    print("\n=== Dependencies (requirements.txt) ===")
    req_path = root / "requirements.txt"
    reqs = parse_requirements(req_path)
    if not reqs:
        warn("requirements.txt empty or unreadable")
    else:
        for r in reqs:
            print(" -", r)

    # Check key packages used by repo
    print("\n=== Import checks (key libs) ===")
    libs = [
        "numpy",
        "pandas",
        "sklearn",
        "xgboost",
        "matplotlib",
        "seaborn",
        "gym",
        "gymnasium",
        "stable_baselines3",
        "torch",
    ]
    for mod in libs:
        ok_imp, out = import_check(mod)
        if ok_imp:
            ok(f"import {mod} -> {out}")
        else:
            warn(f"import {mod} failed -> {out}")

    # 6) Torch GPU situation
    print("\n=== Torch GPU check ===")
    ok_torch, _ = import_check("torch")
    if ok_torch:
        code, out = sh([sys.executable, "-c",
                        "import torch; "
                        "print('torch', torch.__version__); "
                        "print('cuda build', torch.version.cuda); "
                        "print('cuda available', torch.cuda.is_available()); "
                        "print('devices', torch.cuda.device_count() if torch.cuda.is_available() else 0)"])
        print(out if out else "(no output)")
    else:
        warn("torch not installed -> skip GPU check")

    # 7) Repo-specific traps
    print("\n=== Repo-specific checks ===")
    dt = root / "data_training.py"
    if dt.exists() and file_contains(dt, "google.colab"):
        warn("data_training.py imports google.colab -> will crash locally.")
        print("      Fix:\n"
              "      try:\n"
              "          from google.colab import files\n"
              "      except Exception:\n"
              "          files = None\n")

    rl = root / "RL_training.py"
    if rl.exists():
        if file_contains(rl, "import gym") or file_contains(rl, "from gym"):
            ok("RL_training.py uses gym (good if gym installed)")
        if file_contains(rl, "gymnasium"):
            warn("RL_training.py mentions gymnasium -> make sure import matches installed package")

    # 8) Quick “ready to run” verdict
    print("\n=== Verdict ===")
    if missing:
        bad(f"Project NOT ready (missing files: {', '.join(missing)})")
        sys.exit(2)

    # If both numpy and pandas missing, probably deps not installed
    np_ok, _ = import_check("numpy")
    pd_ok, _ = import_check("pandas")
    if not (np_ok and pd_ok):
        warn("Core deps not installed yet. Run:\n"
             "      pip install --upgrade pip\n"
             "      pip install -r requirements.txt")
    else:
        ok("Core deps seem installed.")

    ok("Directory check completed.")
    sys.exit(0)

if __name__ == "__main__":
    main()
