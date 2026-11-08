import shutil
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse

from resilens.api.schemas import RunCreated, RunInfo, RunReport, ScenarioInfo
from resilens.api.worker import RunQueue
from resilens.config.loader import (
    available_modes,
    available_profiles,
    build_run_config,
    load_mode,
    load_profile,
)
from resilens.config.schema import ModeConfig, ProfileConfig, ReplayMode
from resilens.io.validation import SUPPORTED_EXTENSIONS, VideoValidationError, VideoValidator
from resilens.reports.annotated_video import ANNOTATED_VIDEO, render_annotated_video
from resilens.runs.manager import RunManager
from resilens.runs.models import RunNotFoundError, RunRecord, RunStatus

router = APIRouter()

# Artifacts of runs/<run_id>/ the API serves, with their media types
RUN_FILES = {
    "config.json": "application/json",
    "fingerprint.json": "application/json",
    "metrics.json": "application/json",
    "report.json": "application/json",
    "detections.txt": "text/plain",
    "tracks.txt": "text/plain",
    "policy_log.csv": "text/csv",
}
_render_lock = Lock()  # one annotated video is rendered at a time


def get_manager(request: Request) -> RunManager:
    return request.app.state.manager


def get_queue(request: Request) -> RunQueue:
    return request.app.state.queue


def get_uploads_dir(request: Request) -> Path:
    return request.app.state.uploads_dir


def save_upload(video: UploadFile, uploads_dir: Path) -> Path:
    """Store the upload in its own folder under uploads_dir and check it is a readable video."""
    name = Path(video.filename or "").name  # drop any directory part sent by the client
    if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(422, f"Unsupported video format '{Path(name).suffix}'. Supported: {supported}")

    folder = uploads_dir / uuid4().hex
    folder.mkdir(parents=True)
    path = folder / name
    with path.open("wb") as file:
        shutil.copyfileobj(video.file, file)
    try:
        VideoValidator().validate(path)
    except VideoValidationError as error:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(422, f"Invalid video: {error}") from error
    return path


def get_record(manager: RunManager, run_id: str) -> RunRecord:
    try:
        return manager.store.get(run_id)
    except RunNotFoundError as error:
        raise HTTPException(404, f"Run '{run_id}' not found") from error


@router.post("/runs", response_model=RunCreated, status_code=202)
def create_run(
    response: Response,
    mode: str = Form(...),
    profile: str = Form(...),
    video: UploadFile | None = File(None),
    scenario_id: str | None = Form(None),
    replay_mode: ReplayMode = Form(ReplayMode.OFFLINE),
    manager: RunManager = Depends(get_manager),
    queue: RunQueue = Depends(get_queue),
    uploads_dir: Path = Depends(get_uploads_dir),
) -> RunCreated:
    """Start a run on an uploaded video or a prepared scenario; poll GET /runs/{run_id} for progress."""
    if mode not in available_modes():
        raise HTTPException(422, f"Unknown mode '{mode}'. Available: {', '.join(available_modes())}")
    if profile not in available_profiles():
        raise HTTPException(422, f"Unknown profile '{profile}'. Available: {', '.join(available_profiles())}")
    has_video = video is not None and bool(video.filename)
    if has_video == bool(scenario_id):
        raise HTTPException(422, "Provide exactly one of 'video' or 'scenario_id'")
    if scenario_id:
        raise HTTPException(501, "Runs on prepared scenarios need the scenario manifest loader (T-08)")

    source = save_upload(video, uploads_dir)
    try:
        run_id = manager.submit(build_run_config(mode, profile, source, replay_mode=replay_mode))
    except Exception:
        shutil.rmtree(source.parent, ignore_errors=True)
        raise
    queue.submit(run_id)
    response.headers["Location"] = f"/runs/{run_id}"
    return RunCreated(run_id=run_id, status=RunStatus.QUEUED)


@router.get("/runs", response_model=list[RunInfo])
def list_runs(manager: RunManager = Depends(get_manager)) -> list[RunInfo]:
    """All runs, newest first."""
    return [RunInfo.from_record(record) for record in manager.store.list_runs()]


@router.get("/runs/{run_id}", response_model=RunInfo)
def get_run(run_id: str, manager: RunManager = Depends(get_manager)) -> RunInfo:
    return RunInfo.from_record(get_record(manager, run_id))


@router.get("/runs/{run_id}/report", response_model=RunReport)
def get_report(run_id: str, manager: RunManager = Depends(get_manager)) -> RunReport:
    """Metrics of a completed run; 409 while it is queued, running or failed."""
    record = get_record(manager, run_id)
    if record.status != RunStatus.COMPLETED:
        raise HTTPException(409, f"Run '{run_id}' is {record.status}, report is available once it is completed")
    return RunReport(
        run_id=run_id, mode=record.mode, profile=record.profile, metrics=manager.store.load_metrics(run_id)
    )


def get_completed_dir(manager: RunManager, run_id: str) -> Path:
    record = get_record(manager, run_id)
    if record.status != RunStatus.COMPLETED:
        raise HTTPException(409, f"Run '{run_id}' is {record.status}, its results are available once it is completed")
    return manager.store.run_dir(run_id)


@router.get("/runs/{run_id}/files", response_model=list[str])
def list_run_files(run_id: str, manager: RunManager = Depends(get_manager)) -> list[str]:
    """Names of the run artifacts that exist, in the order of RUN_FILES."""
    get_record(manager, run_id)
    run_dir = manager.store.run_dir(run_id)
    return [name for name in RUN_FILES if (run_dir / name).is_file()]


@router.get("/runs/{run_id}/files/{name}")
def get_run_file(run_id: str, name: str, manager: RunManager = Depends(get_manager)) -> FileResponse:
    """One artifact of a run: config snapshot, metrics, MOT detections and tracks, policy log."""
    get_record(manager, run_id)
    path = manager.store.run_dir(run_id) / name
    if name not in RUN_FILES or not path.is_file():
        raise HTTPException(404, f"Run '{run_id}' has no file '{name}'")
    return FileResponse(path, media_type=RUN_FILES[name], filename=name)


@router.get("/runs/{run_id}/video")
def get_annotated_video(run_id: str, manager: RunManager = Depends(get_manager)) -> FileResponse:
    """Source video with the run's tracks and detections drawn on it (WebM), rendered on the first request."""
    run_dir = get_completed_dir(manager, run_id)
    output = run_dir / ANNOTATED_VIDEO
    with _render_lock:
        if not output.is_file():
            if not (run_dir / "tracks.txt").is_file():
                raise HTTPException(404, f"Run '{run_id}' has no tracks to draw")
            source = manager.store.get(run_id).source
            if not source.is_file():
                raise HTTPException(404, f"Source video of run '{run_id}' is no longer available: {source.name}")
            render_annotated_video(
                source, output, run_dir / "detections.txt", run_dir / "tracks.txt", run_dir / "policy_log.csv"
            )
    return FileResponse(output, media_type="video/webm", filename=f"{run_id}.webm")


@router.get("/modes", response_model=list[ModeConfig])
def list_modes() -> list[ModeConfig]:
    """Mode presets A–E from configs/modes/: which stages each one runs."""
    return [load_mode(name) for name in available_modes()]


@router.get("/profiles", response_model=list[ProfileConfig])
def list_profiles() -> list[ProfileConfig]:
    """Resource profiles from configs/profiles/ with their limits."""
    return [load_profile(name) for name in available_profiles()]


@router.get("/scenarios", response_model=list[ScenarioInfo])
async def list_scenarios() -> list[ScenarioInfo]:
    return [ScenarioInfo(scenario_id="mock-fog-1", degradation="fog", level=1, has_annotations=True)]
