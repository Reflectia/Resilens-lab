"""Information that makes a run reproducible and explainable: input hashes, code version, environment.

The code version and the environment are captured once per process: the code a process runs and the libraries it
has loaded do not change while it runs, even if files on disk do. PyTorch and OpenCV are imported only when the
environment is collected, so importing the run store stays cheap.
"""

import functools
import hashlib
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import psutil
from packaging.requirements import InvalidRequirement, Requirement

import resilens

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # the repository, when running from a source checkout
CPU_LOCK = "requirements.lock"
CUDA_LOCK = "requirements-cu130.lock"
MAX_LISTED_CHANGES = 50
CODE_FOLDERS = ("src", "configs")  # new files here are code or configuration a run may use


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_source_checkout() -> bool:
    """True when resilens runs from the repository (editable install), not from a copy in site-packages."""
    return (PROJECT_ROOT / "src" / "resilens").is_dir() and (PROJECT_ROOT / "pyproject.toml").is_file()


def _git(*args: str) -> str | None:
    if not is_source_checkout():
        return None
    try:
        result = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):  # no git installed, as in the Docker image
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_commit() -> str | None:
    """Commit of the running code: RESILENS_GIT_COMMIT when set (the Docker image has no .git), else from git."""
    return os.getenv("RESILENS_GIT_COMMIT") or _git("rev-parse", "HEAD")


def code_version() -> dict:
    """Commit, branch and uncommitted changes: with changes, the commit alone is not the code that ran.

    Changes are edits of tracked files (staged or not) and new files in src/ and configs/ that are not committed yet:
    code and configuration a run may use. `git_diff_sha256` (of the tracked edits) tells apart two runs made from the
    same commit with different uncommitted changes.
    """
    changed = _git("diff", "--name-only", "HEAD")  # None without git
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *CODE_FOLDERS)
    changes = changed.splitlines() if changed else []
    new_files = untracked.splitlines() if untracked else []
    diff = _git("diff", "HEAD") if changes else None
    return {
        "git_commit": git_commit(),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": None if changed is None else bool(changes or new_files),
        "git_changed_files": changes[:MAX_LISTED_CHANGES],
        "git_untracked_files": new_files[:MAX_LISTED_CHANGES],
        "git_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None,
        "package_version": resilens.__version__,
    }


def installed_packages() -> dict[str, str]:
    """Every installed distribution and its exact version, by normalised name."""
    packages = {}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            packages[normalize(name)] = distribution.version
    return dict(sorted(packages.items()))


def normalize(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def lock_file() -> Path | None:
    """The lock file this environment should have been installed from: CUDA lock for a CUDA build of PyTorch."""
    import torch

    path = PROJECT_ROOT / (CUDA_LOCK if torch.version.cuda else CPU_LOCK)
    return path if path.is_file() else None


def lock_differences(lock: Path, packages: dict[str, str]) -> dict[str, dict[str, str]]:
    """Installed packages whose version differs from the lock. Locked packages that are not installed (extras such as
    e2e) and lines for other platforms are left out."""
    differences = {}
    for line in lock.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement:
            continue
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        installed = packages.get(normalize(requirement.name))
        if installed is not None and not requirement.specifier.contains(installed, prereleases=True):
            locked = ",".join(str(spec) for spec in requirement.specifier).removeprefix("==")
            differences[normalize(requirement.name)] = {"locked": locked, "installed": installed}
    return differences


def gpu_info() -> dict:
    """GPUs PyTorch sees, CUDA and cuDNN versions of the PyTorch build, and the NVIDIA driver."""
    import torch

    info: dict = {"torch_cuda": torch.version.cuda, "cudnn": None, "nvidia_driver": None, "gpus": []}
    if not torch.cuda.is_available():
        return info
    info["cudnn"] = torch.backends.cudnn.version()
    info["gpus"] = [
        {
            "name": torch.cuda.get_device_name(index),
            "memory_gb": round(torch.cuda.get_device_properties(index).total_memory / 2**30, 1),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(index))),
        }
        for index in range(torch.cuda.device_count())
    ]
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            driver = pynvml.nvmlSystemGetDriverVersion()
            info["nvidia_driver"] = driver.decode() if isinstance(driver, bytes) else driver
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # NVML missing or not readable: the driver version is then unknown
        pass
    return info


def collect_environment() -> dict:
    """Python, OS, CPU, RAM, GPU / CUDA / cuDNN, key library versions, every installed package and its lock check."""
    import cv2
    import numpy as np
    import torch

    packages = installed_packages()
    lock = lock_file()
    return {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor() or platform.machine(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "torch": torch.__version__,
        **gpu_info(),
        "torchvision": packages.get("torchvision"),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "lock_file": lock.name if lock else None,
        "lock_sha256": file_sha256(lock) if lock else None,
        "differs_from_lock": lock_differences(lock, packages) if lock else {},
        "packages": packages,
    }


@functools.cache
def run_environment() -> dict:
    """Code version and environment of this process, captured on first use."""
    return {
        "captured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code_version(),
        "environment": collect_environment(),
    }
