#!/usr/bin/env python3
"""
Dependency management script for IRP Research Pipeline.
Replaces requirements.txt files.
"""
import subprocess
import sys
import os
import platform

DEPENDENCIES = [
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    "imbalanced-learn>=0.10.0",
    "torch>=1.12.0",
    "pytorch-tabnet>=4.0",
    "shap>=0.41.0",
    "lime>=0.2.0",
    "psutil>=5.9.0",
    "matplotlib>=3.4.0",
    "seaborn>=0.11.0",
    "tqdm>=4.64.0",
    "pytest>=7.0.0",
    "pyarrow",  # For parquet support
]

def check_hardware():
    """Check hardware capabilities."""
    print("--- Hardware Check ---")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")

    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"Total Memory: {mem.total / (1024**3):.2f} GB")
        print(f"Available Memory: {mem.available / (1024**3):.2f} GB")
    except ImportError:
        print("psutil not installed, skipping memory check.")

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("torch not installed yet.")
    print("----------------------\n")

def install_dependencies():
    """Install all required dependencies."""
    print("Installing dependencies...")
    for dep in DEPENDENCIES:
        print(f"Installing {dep}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
    print("All dependencies installed successfully.")

def check_dependencies():
    """Check if all dependencies are installed."""
    missing = []
    for dep in DEPENDENCIES:
        package = dep.split(">=")[0].split("<")[0].replace("-", "_")
        # Special cases for package names vs import names
        if package == "scikit_learn": package = "sklearn"
        if package == "pytorch_tabnet": package = "pytorch_tabnet"
        if package == "imbalanced_learn": package = "imblearn"

        try:
            __import__(package)
        except ImportError:
            missing.append(dep)

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        return False
    else:
        print("All dependencies are satisfied.")
        return True

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "install":
        install_dependencies()

    check_hardware()
    check_dependencies()
