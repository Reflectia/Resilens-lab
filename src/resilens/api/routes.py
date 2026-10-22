from fastapi import APIRouter, File, Form, UploadFile

from resilens.api.schemas import RunCreated, RunInfo, RunReport, ScenarioInfo
from resilens.runs.models import RunStatus

router = APIRouter()

MOCK_RUN_ID = "mock-run-0001"


@router.post("/runs", response_model=RunCreated, status_code=202)
async def create_run(
    mode: str = Form(...),
    profile: str = Form(...),
    video: UploadFile | None = File(None),
    scenario_id: str | None = Form(None),
) -> RunCreated:
    """Start a run on an uploaded video or a prepared scenario."""
    return RunCreated(run_id=MOCK_RUN_ID, status=RunStatus.QUEUED)


@router.get("/runs/{run_id}", response_model=RunInfo)
async def get_run(run_id: str) -> RunInfo:
    return RunInfo(run_id=run_id, status=RunStatus.COMPLETED, mode="A", profile="minimal")


@router.get("/runs/{run_id}/report", response_model=RunReport)
async def get_report(run_id: str) -> RunReport:
    return RunReport(
        run_id=run_id,
        mode="A",
        profile="minimal",
        metrics={"fps": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "peak_ram_mb": 0.0},
    )


@router.get("/scenarios", response_model=list[ScenarioInfo])
async def list_scenarios() -> list[ScenarioInfo]:
    return [ScenarioInfo(scenario_id="mock-fog-1", degradation="fog", level=1, has_annotations=True)]
