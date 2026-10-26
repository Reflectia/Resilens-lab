"""REST API /runs against stub Run Manager and Run Store."""

import os
from collections.abc import Iterator
from pathlib import Path
from threading import Event

import pytest
from fastapi.testclient import TestClient

from resilens.api.app import create_app
from resilens.core.pipeline import Pipeline
from resilens.runs.manager import RunManager
from resilens.testing.runs import STUB_METRICS, StubRunManager


@pytest.fixture
def uploads_dir(tmp_path: Path) -> Path:
    return tmp_path / "uploads"


@pytest.fixture
def client(run_manager: StubRunManager, uploads_dir: Path) -> Iterator[TestClient]:
    with TestClient(create_app(run_manager, uploads_dir)) as test_client:  # runs the lifespan: worker starts
        yield test_client


@pytest.fixture
def video(make_video) -> Path:
    return make_video("clip.mp4", frames=10)


def post_run(client: TestClient, video: Path | None, filename: str | None = None, **form):
    data = {"mode": "A", "profile": "minimal", **form}
    files = {"video": (filename or video.name, video.read_bytes(), "video/mp4")} if video else None
    return client.post("/runs", data=data, files=files)


def wait_for_runs(client: TestClient) -> None:
    client.app.state.queue.join()


def test_create_run_with_video(client, run_manager, video, uploads_dir) -> None:
    response = post_run(client, video)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert response.headers["location"] == f"/runs/{body['run_id']}"

    config = run_manager.store.load_config(body["run_id"])
    assert (config.mode.name, config.profile.name) == ("A", "minimal")
    assert config.source.parent.parent == uploads_dir.resolve()
    assert config.source.read_bytes() == video.read_bytes()


def test_run_is_executed_in_background(client, video) -> None:
    run_id = post_run(client, video).json()["run_id"]
    wait_for_runs(client)

    run = client.get(f"/runs/{run_id}").json()

    assert run["status"] == "completed"
    assert run["finished_at"] is not None
    assert run["error"] is None


def test_report_of_completed_run(client, video) -> None:
    run_id = post_run(client, video).json()["run_id"]
    wait_for_runs(client)

    report = client.get(f"/runs/{run_id}/report")

    assert report.status_code == 200
    assert report.json()["metrics"] == STUB_METRICS
    assert (report.json()["mode"], report.json()["profile"]) == ("A", "minimal")


def test_report_while_running_is_conflict(client, run_manager, video) -> None:
    run_manager.release = Event()
    run_id = post_run(client, video).json()["run_id"]
    assert run_manager.started.wait(timeout=5)

    assert client.get(f"/runs/{run_id}").json()["status"] == "running"
    report = client.get(f"/runs/{run_id}/report")

    assert report.status_code == 409
    assert "running" in report.json()["detail"]
    run_manager.release.set()


def test_failed_run_keeps_error_and_worker_continues(client, run_manager, video) -> None:
    run_manager.fail_with = RuntimeError("decoder crashed")
    failed_id = post_run(client, video).json()["run_id"]
    wait_for_runs(client)
    run_manager.fail_with = None
    next_id = post_run(client, video).json()["run_id"]
    wait_for_runs(client)

    failed = client.get(f"/runs/{failed_id}").json()

    assert (failed["status"], failed["error"]) == ("failed", "decoder crashed")
    assert client.get(f"/runs/{failed_id}/report").status_code == 409
    assert client.get(f"/runs/{next_id}").json()["status"] == "completed"


def test_runs_execute_one_at_a_time_in_order(client, run_manager, video) -> None:
    run_ids = [post_run(client, video).json()["run_id"] for _ in range(3)]
    wait_for_runs(client)

    assert run_manager.executed == run_ids


def test_list_runs_newest_first(client, video) -> None:
    run_ids = [post_run(client, video).json()["run_id"] for _ in range(2)]

    listed = [run["run_id"] for run in client.get("/runs").json()]

    assert listed == run_ids[::-1]


@pytest.mark.parametrize("path", ["/runs/missing", "/runs/missing/report"])
def test_unknown_run_is_not_found(client, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 404
    assert "missing" in response.json()["detail"]


@pytest.mark.parametrize(
    ("form", "message"),
    [
        ({"mode": "Z"}, "Unknown mode 'Z'. Available: A, B, C, D, E"),
        ({"mode": "../profiles/minimal"}, "Unknown mode"),
        ({"profile": "huge"}, "Unknown profile 'huge'. Available: minimal, performance, standard"),
        ({"scenario_id": "MOT20-01/fog_l1"}, "exactly one of 'video' or 'scenario_id'"),
    ],
    ids=["unknown_mode", "mode_path_traversal", "unknown_profile", "video_and_scenario"],
)
def test_invalid_request_is_rejected(client, run_manager, video, uploads_dir, form: dict, message: str) -> None:
    response = post_run(client, video, **form)

    assert response.status_code == 422
    assert message in response.json()["detail"]
    assert run_manager.store.list_runs() == []
    assert not uploads_dir.exists() or not any(uploads_dir.iterdir())


def test_missing_video_and_scenario(client) -> None:
    response = post_run(client, None)

    assert response.status_code == 422
    assert "exactly one" in response.json()["detail"]


def test_missing_mode_field(client, video) -> None:
    response = client.post("/runs", data={"profile": "minimal"}, files={"video": ("clip.mp4", video.read_bytes())})

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("clip.mkv", b"anything", "Unsupported video format '.mkv'"),
        ("clip", b"anything", "Unsupported video format ''"),
        ("broken.mp4", os.urandom(4096), "Invalid video"),
    ],
    ids=["mkv", "no_extension", "corrupted_mp4"],
)
def test_bad_video_is_rejected_and_not_kept(
    client, run_manager, uploads_dir, tmp_path, filename: str, content: bytes, message: str
) -> None:
    upload = tmp_path / "upload.bin"
    upload.write_bytes(content)

    response = post_run(client, upload, filename=filename)

    assert response.status_code == 422
    assert message in response.json()["detail"]
    assert run_manager.store.list_runs() == []
    assert not uploads_dir.exists() or not any(uploads_dir.iterdir())


def test_upload_name_cannot_escape_uploads_dir(client, run_manager, video, uploads_dir) -> None:
    run_id = post_run(client, video, filename="../../evil.mp4").json()["run_id"]

    source = run_manager.store.load_config(run_id).source
    assert source.name == "evil.mp4"
    assert source.parent.parent == uploads_dir.resolve()


def test_scenario_runs_are_not_implemented_yet(client) -> None:
    response = post_run(client, None, scenario_id="MOT20-01/fog_l1")

    assert response.status_code == 501
    assert "T-08" in response.json()["detail"]


def test_list_scenarios(client) -> None:
    assert isinstance(client.get("/scenarios").json(), list)


class FakePipeline:
    def run(self, frames):
        return list(frames)


@pytest.mark.parametrize(("pipeline", "status"), [(None, "failed"), (FakePipeline(), "completed")])
def test_real_run_manager(video, uploads_dir, run_store, monkeypatch, pipeline, status: str) -> None:
    monkeypatch.setattr(Pipeline, "from_config", lambda config: pipeline)
    with TestClient(create_app(RunManager(run_store), uploads_dir)) as client:
        run_id = post_run(client, video).json()["run_id"]
        wait_for_runs(client)
        run = client.get(f"/runs/{run_id}").json()
        report = client.get(f"/runs/{run_id}/report")

    assert run["status"] == status
    if status == "completed":
        assert run["error"] is None
        assert report.json()["metrics"]["processed_frames"] == 10
    else:
        assert "Pipeline is not implemented yet" in run["error"]
        assert report.status_code == 409
