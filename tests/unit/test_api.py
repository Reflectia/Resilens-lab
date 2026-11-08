"""REST API /runs against stub Run Manager and Run Store."""

import os
from collections.abc import Iterator
from pathlib import Path
from threading import Event

import pytest
import torch
from fastapi.testclient import TestClient

from resilens.api.app import create_app
from resilens.core.pipeline import Pipeline
from resilens.detection.yolox_model import yolox_nano
from resilens.enhancement.tanet_model import TANet
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
    assert config.replay_mode == "offline"


def test_create_run_in_realtime(client, run_manager, video) -> None:
    run_id = post_run(client, video, replay_mode="realtime").json()["run_id"]

    assert run_manager.store.load_config(run_id).replay_mode == "realtime"


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


@pytest.mark.parametrize(
    "path", ["/runs/missing", "/runs/missing/report", "/runs/missing/files", "/runs/missing/files/tracks.txt"]
)
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
        ({"replay_mode": "slow"}, "Input should be 'offline' or 'realtime'"),
    ],
    ids=["unknown_mode", "mode_path_traversal", "unknown_profile", "video_and_scenario", "unknown_replay_mode"],
)
def test_invalid_request_is_rejected(client, run_manager, video, uploads_dir, form: dict, message: str) -> None:
    response = post_run(client, video, **form)

    assert response.status_code == 422
    assert message in str(response.json()["detail"])
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
        assert report.json()["metrics"]["frames_dropped"] == 0
    else:
        assert "Pipeline is not implemented yet" in run["error"]
        assert report.status_code == 409


def test_mode_a_runs_real_pipeline(video, uploads_dir, run_store, monkeypatch, tmp_path) -> None:
    """POST /runs -> Pipeline.from_config (bypass, YOLOX-Nano with random weights, flow tracker) -> report."""
    weights = tmp_path / "yolox_nano.pth"
    torch.save({"model": yolox_nano().state_dict()}, weights)
    build = Pipeline.from_config

    def with_test_weights(config):
        config.mode.detector_params["weights_path"] = weights
        return build(config)

    monkeypatch.setattr(Pipeline, "from_config", with_test_weights)
    with TestClient(create_app(RunManager(run_store), uploads_dir)) as client:
        run_id = post_run(client, video).json()["run_id"]
        wait_for_runs(client)
        run = client.get(f"/runs/{run_id}").json()
        report = client.get(f"/runs/{run_id}/report").json()

    assert (run["status"], run["error"]) == ("completed", None)
    assert report["metrics"]["processed_frames"] == 10
    assert {"detections_per_frame", "tracks_per_frame"} <= report["metrics"].keys()
    assert (run_store.run_dir(run_id) / "tracks.txt").is_file()


def test_mode_d_run_saves_policy_log_and_summary(video, uploads_dir, run_store, monkeypatch, tmp_path) -> None:
    yolox, tanet = tmp_path / "yolox_nano.pth", tmp_path / "tanet.pth"
    torch.save({"model": yolox_nano().state_dict()}, yolox)
    torch.save(TANet().state_dict(), tanet)
    build = Pipeline.from_config

    def with_test_weights(config):
        config.mode.detector_params["weights_path"] = yolox
        config.mode.enhancer_params["weights_path"] = tanet
        return build(config)

    monkeypatch.setattr(Pipeline, "from_config", with_test_weights)
    with TestClient(create_app(RunManager(run_store), uploads_dir)) as client:
        run_id = post_run(client, video, mode="D").json()["run_id"]
        wait_for_runs(client)
        run = client.get(f"/runs/{run_id}").json()
        metrics = client.get(f"/runs/{run_id}/report").json()["metrics"]

    assert (run["status"], run["error"]) == ("completed", None)
    log = (run_store.run_dir(run_id) / "policy_log.csv").read_text(encoding="utf-8").splitlines()
    assert len(log) == 1 + 10  # header + one row per frame
    assert {"policy_enhance_share", "policy_downgraded_share", "policy_enhanced_share"} <= metrics.keys()


TRACK_ROW = "1,1,5,5,20,20,0.9,-1,-1,-1\n"


def completed_run(client: TestClient, run_manager: StubRunManager, video: Path, tracks: str | None = None) -> str:
    run_id = post_run(client, video).json()["run_id"]
    wait_for_runs(client)
    if tracks is not None:
        run_dir = run_manager.store.run_dir(run_id)
        (run_dir / "tracks.txt").write_text(tracks)
        (run_dir / "detections.txt").write_text(tracks.replace(",1,", ",-1,"))
    return run_id


def test_run_info_names_the_uploaded_video(client, video) -> None:
    run_id = post_run(client, video, filename="fog_l2.mp4").json()["run_id"]

    assert client.get(f"/runs/{run_id}").json()["source_name"] == "fog_l2.mp4"


def test_run_files_are_listed_and_served(client, run_manager, video) -> None:
    run_id = completed_run(client, run_manager, video, tracks=TRACK_ROW)

    names = client.get(f"/runs/{run_id}/files").json()
    tracks = client.get(f"/runs/{run_id}/files/tracks.txt")
    config = client.get(f"/runs/{run_id}/files/config.json")

    assert names == ["config.json", "fingerprint.json", "metrics.json", "detections.txt", "tracks.txt"]
    assert tracks.status_code == 200 and tracks.text.startswith("1,1,5")
    assert config.json()["mode"]["name"] == "A"


@pytest.mark.parametrize("name", ["record.json", "policy_log.csv", "..%2Frecord.json", "annotated.webm"])
def test_unlisted_or_missing_file_is_not_found(client, run_manager, video, name: str) -> None:
    run_id = completed_run(client, run_manager, video)

    assert client.get(f"/runs/{run_id}/files/{name}").status_code == 404


def test_annotated_video_is_rendered_once(client, run_manager, video, monkeypatch) -> None:
    run_id = completed_run(client, run_manager, video, tracks=TRACK_ROW)

    first = client.get(f"/runs/{run_id}/video")
    monkeypatch.setattr("resilens.api.routes.render_annotated_video", lambda *args: pytest.fail("rendered twice"))
    second = client.get(f"/runs/{run_id}/video")

    assert first.status_code == 200 and first.headers["content-type"] == "video/webm"
    assert first.content[:4] == bytes.fromhex("1a45dfa3")  # EBML header of a WebM file
    assert second.content == first.content


def test_video_needs_a_completed_run_with_tracks_and_source(client, run_manager, video) -> None:
    run_manager.release = Event()
    running_id = post_run(client, video).json()["run_id"]
    assert run_manager.started.wait(timeout=5)
    assert client.get(f"/runs/{running_id}/video").status_code == 409
    run_manager.release.set()
    wait_for_runs(client)
    run_manager.release = None

    no_tracks = client.get(f"/runs/{running_id}/video")
    gone_id = completed_run(client, run_manager, video, tracks=TRACK_ROW)
    run_manager.store.get(gone_id).source.unlink()
    no_source = client.get(f"/runs/{gone_id}/video")

    assert no_tracks.status_code == 404 and "no tracks" in no_tracks.json()["detail"]
    assert no_source.status_code == 404 and "no longer available" in no_source.json()["detail"]


def test_modes_and_profiles_come_from_configs(client) -> None:
    modes = client.get("/modes").json()
    profiles = client.get("/profiles").json()

    assert [mode["name"] for mode in modes] == ["A", "B", "C", "D", "E"]
    assert modes[3]["use_policy"] is True and modes[0]["detector"] == "yolox_nano"
    assert {profile["name"] for profile in profiles} == {"minimal", "standard", "performance"}
    assert "frame_budget_ms" in profiles[0]
