"""Single entry point used by CLI and REST API to start and execute runs.

The REST API (T-16) calls submit() in the request and execute() later in a background worker,
one run at a time; it reads runs through `store`. Keep these signatures when implementing (T-06).
"""

from resilens.config.schema import RunConfig
from resilens.runs.store import RunStore


class RunManager:
    def __init__(self, store: RunStore) -> None:
        self.store = store

    def submit(self, config: RunConfig) -> str:
        """Register a new run (status QUEUED) and return its run_id. Must return quickly: no processing here."""
        raise NotImplementedError("Run Manager is not implemented yet (T-06)")

    def execute(self, run_id: str) -> None:
        """Validate input, build the pipeline, process the video, compute metrics, write report.

        Sets RUNNING at the start, then COMPLETED after save_metrics, or FAILED with the error message.
        If it raises anyway, the worker marks the run FAILED itself.
        """
        raise NotImplementedError("Run Manager is not implemented yet (T-06)")

    def compare_with_previous(self, run_id: str) -> dict:
        raise NotImplementedError("Run Manager is not implemented yet (T-06)")
