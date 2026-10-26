from pathlib import Path

import pytest

from resilens.config.schema import ModeConfig, ProfileConfig, RunConfig
from resilens.runs.models import RunNotFoundError, RunStatus
from resilens.runs.store import RunStore


def config_for(source: Path) -> RunConfig:
    return RunConfig(mode=ModeConfig(name="A"), profile=ProfileConfig(name="minimal"), source=source, seed=17)


def test_run_persists_config_status_and_metrics_after_reopening(tmp_path, make_video) -> None:
    video = make_video()
    root = tmp_path / "runs"
    store = RunStore(root)
    record = store.create_run(config_for(video))
    store.update_status(record.run_id, RunStatus.RUNNING)
    store.save_metrics(record.run_id, {"fps": 12.5})
    store.update_status(record.run_id, RunStatus.COMPLETED)

    reopened = RunStore(root)
    saved = reopened.get(record.run_id)
    assert saved.status == RunStatus.COMPLETED
    assert saved.finished_at is not None
    assert reopened.load_config(record.run_id).seed == 17
    assert reopened.load_config(record.run_id).source == video.resolve()
    assert reopened.load_metrics(record.run_id) == {"fps": 12.5}
    assert reopened.run_dir(record.run_id) == root / record.run_id
    assert video.is_file()


def test_missing_and_changed_source_are_rejected(tmp_path, make_video) -> None:
    video = make_video()
    store = RunStore(tmp_path / "runs")
    run_id = store.create_run(config_for(video)).run_id
    store.verify_source(run_id)

    with video.open("ab") as source:
        source.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        store.verify_source(run_id)

    video.unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        store.verify_source(run_id)


def test_recovery_marks_only_incomplete_runs_failed(tmp_path, make_video) -> None:
    video = make_video()
    store = RunStore(tmp_path / "runs")
    queued = store.create_run(config_for(video)).run_id
    running = store.create_run(config_for(video)).run_id
    completed = store.create_run(config_for(video)).run_id
    store.update_status(running, RunStatus.RUNNING)
    store.update_status(completed, RunStatus.RUNNING)
    store.update_status(completed, RunStatus.COMPLETED)

    RunStore(store.root).recover_interrupted()

    assert store.get(queued).status == RunStatus.FAILED
    assert store.get(running).status == RunStatus.FAILED
    assert "restart" in store.get(running).error
    assert store.get(completed).status == RunStatus.COMPLETED
    assert all(store.get(run_id).finished_at is not None for run_id in (queued, running, completed))


def test_unknown_run_and_invalid_transition(tmp_path, make_video) -> None:
    store = RunStore(tmp_path / "runs")
    with pytest.raises(RunNotFoundError):
        store.get("../outside")
    with pytest.raises(RunNotFoundError):
        store.load_metrics("0" * 32)

    run_id = store.create_run(config_for(make_video())).run_id
    assert store.load_metrics(run_id) == {}
    with pytest.raises(ValueError, match="Invalid run status transition"):
        store.update_status(run_id, RunStatus.COMPLETED)


def test_list_runs_is_newest_first(tmp_path, make_video) -> None:
    store = RunStore(tmp_path / "runs")
    video = make_video()
    first = store.create_run(config_for(video)).run_id
    second = store.create_run(config_for(video)).run_id

    assert [record.run_id for record in RunStore(store.root).list_runs()] == [second, first]
