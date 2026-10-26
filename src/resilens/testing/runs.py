"""Run Manager for API tests: real Run Store, execute() fakes processing with fixed metrics."""

from threading import Event

from resilens.runs.manager import RunManager
from resilens.runs.models import RunStatus
from resilens.runs.store import RunStore

STUB_METRICS = {"processing_time_s": 0.4, "fps": 25.0, "peak_ram_mb": 512.0, "processed_frames": 10.0}


class StubRunManager(RunManager):
    """Set `fail_with` to make execute() raise, or `release` to an Event to hold execute() until it is set."""

    def __init__(self, store: RunStore) -> None:
        super().__init__(store)
        self.executed: list[str] = []
        self.fail_with: Exception | None = None
        self.release: Event | None = None
        self.started = Event()

    def execute(self, run_id: str) -> None:
        self.store.update_status(run_id, RunStatus.RUNNING)
        self.started.set()
        if self.release is not None:
            self.release.wait(timeout=5)
        self.executed.append(run_id)
        if self.fail_with is not None:
            raise self.fail_with
        self.store.save_metrics(run_id, STUB_METRICS)
        self.store.update_status(run_id, RunStatus.COMPLETED)
