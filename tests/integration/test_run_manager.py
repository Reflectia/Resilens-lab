import json

from resilens import cli
from resilens.config.schema import ModeConfig, ProfileConfig, RunConfig
from resilens.core.pipeline import Pipeline
from resilens.runs.manager import RunManager
from resilens.runs.models import RunStatus
from resilens.runs.store import RunStore
from resilens.testing.mocks import MockDetector, MockTracker


class TestPipeline:
    def run(self, frames):
        detector = MockDetector()
        tracker = MockTracker()
        return [tracker.process(detector.process(frame)) for frame in frames]


def test_run_creates_consistent_report_for_cli_and_api_contract(tmp_path, make_video, monkeypatch, capsys) -> None:
    video = make_video(frames=5)
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store)
    config = RunConfig(mode=ModeConfig(name="A"), profile=ProfileConfig(name="minimal"), source=video)
    monkeypatch.setattr(Pipeline, "from_config", lambda config: TestPipeline())
    run_id = manager.submit(config)

    assert store.get(run_id).status == RunStatus.QUEUED
    manager.execute(run_id)

    record = store.get(run_id)
    assert record.status == RunStatus.COMPLETED
    assert record.finished_at is not None
    metrics = store.load_metrics(run_id)
    report = store.load_report(run_id)
    assert metrics["processed_frames"] == 5
    assert metrics["processing_time_s"] > 0
    assert metrics["fps"] == metrics["processed_frames"] / metrics["processing_time_s"]
    assert metrics["peak_ram_mb"] > 0
    assert report["metrics"] == metrics
    assert report["run_id"] == run_id

    monkeypatch.setattr(cli, "RUNS_DIR", store.root)
    assert cli.main(["report", run_id]) is None
    assert json.loads(capsys.readouterr().out) == report


def test_missing_pipeline_is_a_failed_run_without_report(tmp_path, make_video, monkeypatch) -> None:
    video = make_video()
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store)
    run_id = manager.submit(RunConfig(mode=ModeConfig(name="A"), profile=ProfileConfig(name="minimal"), source=video))
    monkeypatch.setattr(Pipeline, "from_config", lambda config: None)

    manager.execute(run_id)

    record = store.get(run_id)
    assert record.status == RunStatus.FAILED
    assert "Pipeline" in record.error
    assert record.finished_at is not None
    assert not (store.run_dir(run_id) / "report.json").exists()


def test_changed_source_fails_before_pipeline_runs(tmp_path, make_video, monkeypatch) -> None:
    video = make_video()
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store)
    run_id = manager.submit(RunConfig(mode=ModeConfig(name="A"), profile=ProfileConfig(name="minimal"), source=video))
    with video.open("ab") as source:
        source.write(b"changed")
    monkeypatch.setattr(Pipeline, "from_config", lambda config: TestPipeline())

    manager.execute(run_id)

    assert store.get(run_id).status == RunStatus.FAILED
    assert "changed" in store.get(run_id).error


def test_manager_marks_orphaned_runs_failed_on_startup(tmp_path, make_video) -> None:
    video = make_video()
    store = RunStore(tmp_path / "runs")
    orphan = store.create_run(RunConfig(mode=ModeConfig(name="A"), profile=ProfileConfig(name="minimal"), source=video))

    RunManager(RunStore(store.root))

    assert store.get(orphan.run_id).status == RunStatus.FAILED
    assert "restart" in store.get(orphan.run_id).error
