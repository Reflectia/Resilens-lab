from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RunRecord:
    run_id: str
    status: RunStatus
    mode: str
    profile: str
    source: Path
    created_at: datetime
    finished_at: datetime | None = None
    error: str | None = None
