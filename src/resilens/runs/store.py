from pathlib import Path

from resilens.config.schema import RunConfig
from resilens.runs.models import RunRecord, RunStatus


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def create_run(self, config: RunConfig) -> RunRecord:
        """Allocate run_id, create the folder, snapshot config and fingerprint."""
        pass

    def run_dir(self, run_id: str) -> Path:
        pass

    def update_status(self, run_id: str, status: RunStatus, error: str | None = None) -> None:
        pass

    def save_metrics(self, run_id: str, metrics: dict) -> None:
        pass

    def get(self, run_id: str) -> RunRecord:
        pass

    def list_runs(self) -> list[RunRecord]:
        pass
