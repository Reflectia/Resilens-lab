"""Write or check the dependency lock files.

    python scripts/lock_dependencies.py           # update the locks after changing dependencies in pyproject.toml
    python scripts/lock_dependencies.py --check   # fail if a lock does not match pyproject.toml (CI)

The two locks differ only in the PyTorch build and the CUDA runtime packages it needs:

- requirements.lock: CPU build of PyTorch, used by CI, the Docker image and laptops;
- requirements-cu130.lock: CUDA 13.0 build for the performance profile (NVIDIA driver 580 or newer).
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKS = {"requirements.lock": "cpu", "requirements-cu130.lock": "cu130"}  # lock file -> PyTorch backend
PYTHON_VERSION = "3.13"  # the oldest Python pyproject.toml allows


def compile_lock(backend: str, output: Path, extra_args: list[str]) -> None:
    command = [
        sys.executable, "-m", "uv", "pip", "compile", "pyproject.toml", "--all-extras", "--universal",
        "--python-version", PYTHON_VERSION, "--torch-backend", backend, "--quiet",
        "--custom-compile-command", "python scripts/lock_dependencies.py", "-o", str(output), *extra_args,
    ]  # fmt: skip
    subprocess.run(command, cwd=ROOT, check=True)


def read(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.is_file() else []


def main() -> int:
    parser = argparse.ArgumentParser(description="Write or check the dependency lock files.")
    parser.add_argument("--check", action="store_true", help="only check that the locks match pyproject.toml")
    args, extra_args = parser.parse_known_args()

    outdated = []
    with tempfile.TemporaryDirectory() as folder:
        for lock, backend in LOCKS.items():
            current = ROOT / lock
            candidate = Path(folder) / lock
            if current.is_file():  # its versions are the preferences of the new resolution
                candidate.write_text(current.read_text(encoding="utf-8"), encoding="utf-8")
            compile_lock(backend, candidate, extra_args)
            if read(candidate) == read(current):
                continue
            outdated.append(lock)
            if not args.check:
                current.write_text("\n".join(read(candidate)) + "\n", encoding="utf-8")

    if args.check and outdated:
        print(f"Out of date: {', '.join(outdated)}. Run: python scripts/lock_dependencies.py", file=sys.stderr)
        return 1
    print(f"Updated: {', '.join(outdated)}" if outdated else "Lock files are up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
