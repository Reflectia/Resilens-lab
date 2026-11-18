from datetime import datetime

from pydantic import BaseModel

from resilens.config.devices import device_name, device_unavailable_reason
from resilens.config.schema import ProfileConfig
from resilens.runs.models import RunRecord, RunStatus


class RunCreated(BaseModel):
    run_id: str
    status: RunStatus


class RunInfo(BaseModel):
    run_id: str
    status: RunStatus
    mode: str
    profile: str
    source_name: str
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
            source_name=record.source.name,
            created_at=record.created_at,
            finished_at=record.finished_at,
            error=record.error,
        )


class ProfileInfo(ProfileConfig):
    """A resource profile and whether the API host has its device (a GPU profile needs CUDA)."""

    available: bool
    device_name: str | None = None  # the GPU a CUDA profile runs on
    unavailable_reason: str | None = None

    @classmethod
    def from_profile(cls, profile: ProfileConfig) -> "ProfileInfo":
        reason = device_unavailable_reason(profile.device)
        return cls(
            **profile.model_dump(),
            available=reason is None,
            device_name=device_name(profile.device),
            unavailable_reason=reason,
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
