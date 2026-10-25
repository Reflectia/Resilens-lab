"""Run Store (T-06): one folder per run under runs/<run_id>/.

The REST API (T-16) relies on the contract in these docstrings; keep the signatures, statuses and
exceptions when implementing. Status flow: QUEUED -> RUNNING -> COMPLETED | FAILED.
Methods may be called from the API thread and the run worker thread at the same time.
Remove this second comment after implementing store.
"""

from pathlib import Path

from resilens.config.schema import RunConfig
from resilens.runs.models import RunRecord, RunStatus


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def create_run(self, config: RunConfig) -> RunRecord:
        """Allocate run_id, create the folder, snapshot config and fingerprint.

        Returns the new record with status QUEUED and created_at set.
        """
        raise NotImplementedError("Run Store is not implemented yet (T-06)")

    def run_dir(self, run_id: str) -> Path:
        raise NotImplementedError("Run Store is not implemented yet (T-06)")

    def update_status(self, run_id: str, status: RunStatus, error: str | None = None) -> None:
        """Set the status; COMPLETED and FAILED also set finished_at. Raises RunNotFoundError."""
        raise NotImplementedError("Run Store is not implemented yet (T-06)")

    def save_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        raise NotImplementedError("Run Store is not implemented yet (T-06)")

    def load_metrics(self, run_id: str) -> dict[str, float]:
        """Metrics saved by save_metrics, e.g. fps, p50_ms, p95_ms, peak_ram_mb.

        Raises RunNotFoundError for an unknown run; returns {} if no metrics were saved.
        """
        raise NotImplementedError("Run Store is not implemented yet (T-06)")

    def get(self, run_id: str) -> RunRecord:
        """Raises RunNotFoundError for an unknown run_id."""
        raise NotImplementedError("Run Store is not implemented yet (T-06)")

    def list_runs(self) -> list[RunRecord]:
        """All runs, newest first."""
        raise NotImplementedError("Run Store is not implemented yet (T-06)")
