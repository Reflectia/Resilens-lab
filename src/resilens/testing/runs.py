"""In-memory Run Store and Run Manager that follow the T-06 contract (runs/store.py, runs/manager.py).

Needs T-06 implementation.
"""

from datetime import datetime
from pathlib import Path
from threading import Event, Lock

from resilens.config.schema import RunConfig
from resilens.runs.manager import RunManager
from resilens.runs.models import RunNotFoundError, RunRecord, RunStatus
from resilens.runs.store import RunStore

STUB_METRICS = {"fps": 25.0, "p50_ms": 30.0, "p95_ms": 45.0, "peak_ram_mb": 512.0}


class StubRunStore(RunStore):
    """In-memory Run Store following the RunStore contract, until T-06 is implemented."""

    def __init__(self) -> None:
        super().__init__(Path("unused"))
        self.records: dict[str, RunRecord] = {}
        self.configs: dict[str, RunConfig] = {}
        self.metrics: dict[str, dict[str, float]] = {}
        self._lock = Lock()

    def create_run(self, config: RunConfig) -> RunRecord:
        with self._lock:
            run_id = f"run-{len(self.records) + 1:04d}"
            record = RunRecord(
                run_id, RunStatus.QUEUED, config.mode.name, config.profile.name, config.source, datetime.now()
            )
            self.records[run_id], self.configs[run_id] = record, config
            return record

    def update_status(self, run_id: str, status: RunStatus, error: str | None = None) -> None:
        record = self.get(run_id)
        record.status, record.error = status, error
        if status in (RunStatus.COMPLETED, RunStatus.FAILED):
            record.finished_at = datetime.now()

    def save_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        self.get(run_id)
        self.metrics[run_id] = metrics

    def load_metrics(self, run_id: str) -> dict[str, float]:
        self.get(run_id)
        return self.metrics.get(run_id, {})

    def get(self, run_id: str) -> RunRecord:
        try:
            return self.records[run_id]
        except KeyError:
            raise RunNotFoundError(run_id) from None

    def list_runs(self) -> list[RunRecord]:
        return sorted(self.records.values(), key=lambda record: record.created_at, reverse=True)


class StubRunManager(RunManager):
    """Run Manager following the RunManager contract; execute() fakes processing with fixed metrics.

    Set `fail_with` to make execute() raise, or `release` to an Event to hold execute() until it is set.
    """

    def __init__(self, store: StubRunStore) -> None:
        super().__init__(store)
        self.executed: list[str] = []
        self.fail_with: Exception | None = None
        self.release: Event | None = None
        self.started = Event()

    def submit(self, config: RunConfig) -> str:
        return self.store.create_run(config).run_id

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
