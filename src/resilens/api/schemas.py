from pydantic import BaseModel

from resilens.runs.models import RunStatus


class RunCreated(BaseModel):
    run_id: str
    status: RunStatus


class RunInfo(BaseModel):
    run_id: str
    status: RunStatus
    mode: str
    profile: str
    error: str | None = None


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
