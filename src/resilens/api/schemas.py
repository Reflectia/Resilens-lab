from datetime import datetime

from pydantic import BaseModel

from resilens.runs.models import RunRecord, RunStatus


class RunCreated(BaseModel):
    run_id: str
    status: RunStatus


class RunInfo(BaseModel):
    run_id: str
    status: RunStatus
    mode: str
    profile: str
    created_at: datetime
    finished_at: datetime | None = None
    error: str | None = None

    @classmethod
    def from_record(cls, record: RunRecord) -> "RunInfo":
        return cls(
            run_id=record.run_id,
            status=record.status,
            mode=record.mode,
            profile=record.profile,
            created_at=record.created_at,
            finished_at=record.finished_at,
            error=record.error,
        )


class RunReport(BaseModel):
    run_id: str
    mode: str
    profile: str
    metrics: dict[str, float]
    report_url: str | None = None


class ScenarioInfo(BaseModel):
    scenario_id: str
    degradation: str
    level: int
    has_annotations: bool
