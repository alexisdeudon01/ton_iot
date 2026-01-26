from __future__ import annotations

import platform
import shutil
import subprocess
from typing import Dict, Tuple


def _run(cmd: list[str]) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except Exception as exc:
        return 1, "", str(exc)


def _install_instructions(os_name: str) -> str:
    if os_name == "windows":
        return (
            "Windows: installe Docker Desktop, active WSL2 et redémarre l’ordinateur. "
            "Ensuite relance Docker Desktop."
        )
    if os_name == "darwin":
        return "macOS: installe Docker Desktop et démarre l’application Docker."
    return (
        "Linux: installe Docker Engine + Docker Compose plugin + NVIDIA Container Toolkit. "
        "Ajoute ton utilisateur au groupe docker et redémarre la session."
    )


def check_docker_environment(print_report: bool = True) -> Dict[str, object]:
    os_name = platform.system().lower()
    status: Dict[str, object] = {
        "os": os_name,
        "docker_cli": False,
        "docker_version": None,
        "buildx": False,
        "buildx_version": None,
        "compose": False,
        "compose_version": None,
        "daemon_access": False,
        "daemon_error": None,
        "nvidia_runtime": False,
        "nvidia_smi": False,
        "docker_ready": False,
        "gpu_ready": False,
    }

    if shutil.which("docker") is None:
        if print_report:
            print("[docker] Docker client introuvable.")
            print(f"[docker] { _install_instructions(os_name) }")
        return status

    status["docker_cli"] = True
    rc, out, err = _run(["docker", "--version"])
    if rc == 0:
        status["docker_version"] = out
    else:
        status["docker_version"] = err or out

    rc, out, err = _run(["docker", "buildx", "version"])
    if rc == 0:
        status["buildx"] = True
        status["buildx_version"] = out
    else:
        status["buildx_version"] = err or out

    rc, out, err = _run(["docker", "compose", "version"])
    if rc == 0:
        status["compose"] = True
        status["compose_version"] = out
    else:
        rc2, out2, err2 = _run(["docker-compose", "version"])
        if rc2 == 0:
            status["compose"] = True
            status["compose_version"] = out2
        else:
            status["compose_version"] = err2 or out2

    rc, out, err = _run(["docker", "info", "--format", "{{json .Runtimes}}"])
    if rc == 0:
        status["daemon_access"] = True
        status["nvidia_runtime"] = "nvidia" in out.lower()
    else:
        status["daemon_error"] = err or out

    status["nvidia_smi"] = shutil.which("nvidia-smi") is not None
    status["docker_ready"] = bool(status["docker_cli"] and status["buildx"] and status["compose"] and status["daemon_access"])
    status["gpu_ready"] = bool(status["nvidia_smi"] and status["nvidia_runtime"])

    if print_report:
        print(f"[docker] client: {status['docker_version']}")
        print(f"[docker] buildx: {status['buildx_version']}")
        print(f"[docker] compose: {status['compose_version']}")
        if status["daemon_access"]:
            print("[docker] daemon: OK")
        else:
            print(f"[docker] daemon: KO ({status['daemon_error']})")
            print(f"[docker] { _install_instructions(os_name) }")
        print(f"[docker] GPU runtime: {'OK' if status['gpu_ready'] else 'KO'}")

    return status


if __name__ == "__main__":
    check_docker_environment(print_report=True)
