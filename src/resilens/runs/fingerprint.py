"""Information that makes a run reproducible: code version, input hashes, environment."""

from pathlib import Path


def git_commit() -> str | None:
    pass


def file_sha256(path: Path) -> str:
    pass


def collect_environment() -> dict:
    """Python / package versions, OS, CPU, GPU and CUDA info."""
    pass
