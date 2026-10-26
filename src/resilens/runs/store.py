"""Persistent run records and artifacts shared by CLI and REST API."""

import json
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from uuid import uuid4

from resilens.config.schema import RunConfig
from resilens.runs.fingerprint import file_sha256
from resilens.runs.models import RunNotFoundError, RunRecord, RunStatus


def _write_json(path: Path, value: dict) -> None:
    """Replace a complete JSON artifact so readers never see a partial write."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._lock = RLock()

    def create_run(self, config: RunConfig) -> RunRecord:
        """Register a queued run with an immutable config and input fingerprint."""
        source = config.source.resolve(strict=True)
        snapshot = config.model_copy(update={"source": source})
        fingerprint = file_sha256(source)
        record = RunRecord(
            run_id=uuid4().hex,
            status=RunStatus.QUEUED,
            mode=config.mode.name,
            profile=config.profile.name,
            source=source,
            created_at=datetime.now(UTC),
        )
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            directory = self.root / record.run_id
            directory.mkdir()
            _write_json(directory / "config.json", snapshot.model_dump(mode="json"))
            _write_json(directory / "fingerprint.json", {"source_sha256": fingerprint})
            self._save_record(record)
        return record

    def run_dir(self, run_id: str) -> Path:
        with self._lock:
            directory = self._directory(run_id)
            if not (directory / "record.json").is_file():
                raise RunNotFoundError(run_id)
            return directory

    def update_status(self, run_id: str, status: RunStatus, error: str | None = None) -> None:
        with self._lock:
            record = self.get(run_id)
            allowed = {
                RunStatus.QUEUED: {RunStatus.RUNNING, RunStatus.FAILED},
                RunStatus.RUNNING: {RunStatus.COMPLETED, RunStatus.FAILED},
            }
            if status not in allowed.get(record.status, set()):
                raise ValueError(f"Invalid run status transition: {record.status} -> {status}")
            record.status = status
            record.error = error if status == RunStatus.FAILED else None
            record.finished_at = datetime.now(UTC) if status in (RunStatus.COMPLETED, RunStatus.FAILED) else None
            self._save_record(record)

    def save_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        with self._lock:
            directory = self.run_dir(run_id)
            _write_json(directory / "metrics.json", metrics)

    def load_metrics(self, run_id: str) -> dict[str, float]:
        with self._lock:
            path = self.run_dir(run_id) / "metrics.json"
            return _read_json(path) if path.is_file() else {}

    def save_report(self, run_id: str, report: dict) -> Path:
        with self._lock:
            path = self.run_dir(run_id) / "report.json"
            _write_json(path, report)
            return path

    def load_report(self, run_id: str) -> dict:
        with self._lock:
            path = self.run_dir(run_id) / "report.json"
            return _read_json(path)

    def load_config(self, run_id: str) -> RunConfig:
        with self._lock:
            path = self.run_dir(run_id) / "config.json"
            return RunConfig.model_validate_json(path.read_text(encoding="utf-8"))

    def verify_source(self, run_id: str) -> None:
        """Reject a missing or changed source before executing a saved run."""
        with self._lock:
            directory = self.run_dir(run_id)
            source = self.load_config(run_id).source
            expected = _read_json(directory / "fingerprint.json")["source_sha256"]
        if not source.is_file():
            raise FileNotFoundError(f"Run source video is missing: {source}")
        if file_sha256(source) != expected:
            raise ValueError(f"Run source video changed after submission: {source}")

    def get(self, run_id: str) -> RunRecord:
        with self._lock:
            path = self._directory(run_id) / "record.json"
            if not path.is_file():
                raise RunNotFoundError(run_id)
            value = _read_json(path)
            return RunRecord(
                run_id=value["run_id"],
                status=RunStatus(value["status"]),
                mode=value["mode"],
                profile=value["profile"],
                source=Path(value["source"]),
                created_at=datetime.fromisoformat(value["created_at"]),
                finished_at=datetime.fromisoformat(value["finished_at"]) if value["finished_at"] else None,
                error=value["error"],
            )

    def list_runs(self) -> list[RunRecord]:
        with self._lock:
            if not self.root.is_dir():
                return []
            records = [self.get(path.name) for path in self.root.iterdir() if (path / "record.json").is_file()]
            return sorted(records, key=lambda record: record.created_at, reverse=True)

    def recover_interrupted(self) -> None:
        """Mark runs orphaned by a prior process as failed on service startup."""
        with self._lock:
            for record in self.list_runs():
                if record.status in (RunStatus.QUEUED, RunStatus.RUNNING):
                    self.update_status(record.run_id, RunStatus.FAILED, "Run interrupted by service restart")

    def _directory(self, run_id: str) -> Path:
        if len(run_id) != 32 or any(character not in "0123456789abcdef" for character in run_id):
            raise RunNotFoundError(run_id)
        return self.root / run_id

    def _save_record(self, record: RunRecord) -> None:
        value = {
            "run_id": record.run_id,
            "status": record.status.value,
            "mode": record.mode,
            "profile": record.profile,
            "source": str(record.source),
            "created_at": record.created_at.isoformat(),
            "finished_at": record.finished_at.isoformat() if record.finished_at else None,
            "error": record.error,
        }
        _write_json(self.root / record.run_id / "record.json", value)
