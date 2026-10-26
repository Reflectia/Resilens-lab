"""Information that makes a run reproducible: code version, input hashes, environment."""

import hashlib
from pathlib import Path


def git_commit() -> str | None:
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_environment() -> dict:
    """Python / package versions, OS, CPU, GPU and CUDA info."""
    pass
