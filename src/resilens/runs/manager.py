"""Single entry point used by CLI and REST API to start and execute runs."""

from resilens.config.schema import RunConfig
from resilens.runs.store import RunStore


class RunManager:
    def __init__(self, store: RunStore) -> None:
        self.store = store

    def submit(self, config: RunConfig) -> str:
        """Register a new run and return its run_id."""
        pass

    def execute(self, run_id: str) -> None:
        """Validate input, build the pipeline, process the video, compute metrics, write report."""
        pass

    def compare_with_previous(self, run_id: str) -> dict:
        pass
