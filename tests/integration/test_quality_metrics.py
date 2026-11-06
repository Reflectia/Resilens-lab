import json

import pytest

from resilens import cli
from resilens.config.schema import ModeConfig, ProfileConfig, RunConfig
from resilens.core.contracts import Detection, Track
from resilens.core.pipeline import Pipeline
from resilens.runs.manager import RunManager
from resilens.runs.models import RunStatus
from resilens.runs.store import RunStore
from resilens.testing.mocks import MockDetector, MockTracker


class FixedPipeline:
    def run(self, frames):
        detector, tracker = MockDetector(), MockTracker()
        return [tracker.process(detector.process(frame)) for frame in frames]


class SwitchedPipeline(FixedPipeline):
    def run(self, frames):
        frames = super().run(frames)
        frames[1].tracks = [Track(track_id=2, bbox=(10.0, 10.0, 30.0, 40.0), score=0, class_id=0)]
        return frames


class CrowdedPipeline(FixedPipeline):
    def run(self, frames):
        frames = super().run(frames)
        frames[0].detections.extend(Detection((0, 0, 2, 2), 0.8, 0) for _ in range(100))
        return frames


class EmptyPipeline:
    def run(self, frames):
        return list(frames)


def test_annotated_run_reports_perfect_detection_and_tracking(tmp_path, make_video, monkeypatch):
    video = make_video(frames=2)
    coco = tmp_path / "ground_truth.json"
    mot = tmp_path / "ground_truth.txt"
    coco.write_text(
        json.dumps(
            {
                "images": [{"id": i, "file_name": f"{i:06}.jpg", "width": 64, "height": 48} for i in (1, 2)],
                "categories": [{"id": 1, "name": "person"}],
                "annotations": [
                    {"id": i, "image_id": i, "category_id": 1, "bbox": [10, 10, 20, 30], "area": 600, "iscrowd": 0}
                    for i in (1, 2)
                ],
            }
        ),
        encoding="utf-8",
    )
    mot.write_text("1,1,10,10,20,30,1,1,1\n2,1,10,10,20,30,1,1,1\n", encoding="utf-8")
    monkeypatch.setattr(Pipeline, "from_config", lambda config: FixedPipeline())
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store)
    config = RunConfig(
        mode=ModeConfig(name="A"),
        profile=ProfileConfig(name="minimal"),
        source=video,
        annotations=coco,
        tracking_annotations=mot,
    )
    run_id = manager.submit(config)
    manager.execute(run_id)
    assert store.get(run_id).status == RunStatus.COMPLETED, store.get(run_id).error
    metrics = store.load_metrics(run_id)
    for name in ("map_50_95", "precision", "recall", "hota", "idf1"):
        assert metrics[name] == pytest.approx(1.0)
    assert metrics["id_switches"] == 0
    other = manager.submit(config.model_copy(update={"mode": ModeConfig(name="B")}))
    manager.execute(other)
    monkeypatch.setattr(cli, "RUNS_DIR", store.root)
    output = tmp_path / "comparison.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "test_cases": [
                    {"file": video.name, "distortion": "none", "level": 0, "gt_coco": coco.name, "gt_mot": mot.name}
                ]
            }
        )
    )
    cli.main(["compare", run_id, other, "--output", str(output), "--manifest", str(manifest)])
    comparison = json.loads(output.read_text(encoding="utf-8"))
    assert comparison["scenarios"][0]["A"]["run_id"] == run_id
    assert comparison["scenarios"][0]["B"]["run_id"] == other
    assert comparison["overall"]["A"]["hota"] == pytest.approx(1)
    assert comparison["overall"]["B"]["map_50_95"] == pytest.approx(1)
    assert comparison["groups"][0]["metrics"]["A"]["hota"] == pytest.approx(1)
    mot.write_text(mot.read_text() + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.main(["compare", run_id, other, "--output", str(output), "--manifest", str(manifest)])


def test_tracking_id_switch_and_zero_score_are_evaluated(tmp_path, make_video, monkeypatch):
    video = make_video(frames=2)
    coco = tmp_path / "ground_truth.json"
    mot = tmp_path / "ground_truth.txt"
    coco.write_text(
        json.dumps(
            {
                "images": [{"id": i, "width": 64, "height": 48} for i in (1, 2)],
                "categories": [{"id": 1, "name": "person"}],
                "annotations": [
                    {"id": i, "image_id": i, "category_id": 1, "bbox": [10, 10, 20, 30], "area": 600, "iscrowd": 0}
                    for i in (1, 2)
                ],
            }
        ),
        encoding="utf-8",
    )
    mot.write_text("1,1,10,10,20,30,1,1,1\n2,1,10,10,20,30,1,1,1\n", encoding="utf-8")
    monkeypatch.setattr(Pipeline, "from_config", lambda config: SwitchedPipeline())
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store)
    config = RunConfig(
        mode=ModeConfig(name="A"),
        profile=ProfileConfig(name="minimal"),
        source=video,
        annotations=coco,
        tracking_annotations=mot,
    )
    run_id = manager.submit(config)
    manager.execute(run_id)
    assert store.get(run_id).status == RunStatus.COMPLETED, store.get(run_id).error
    metrics = store.load_metrics(run_id)
    assert metrics["id_switches"] == 1
    assert metrics["idf1"] == pytest.approx(0.5)


def test_run_fails_when_only_one_annotation_is_provided(tmp_path, make_video, monkeypatch):
    video = make_video(frames=2)
    monkeypatch.setattr(Pipeline, "from_config", lambda config: FixedPipeline())
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store)
    config = RunConfig(
        mode=ModeConfig(name="A"),
        profile=ProfileConfig(name="minimal"),
        source=video,
        annotations=tmp_path / "missing.json",
    )
    run_id = manager.submit(config)
    manager.execute(run_id)
    assert store.get(run_id).status == RunStatus.FAILED
    assert "annotation" in store.get(run_id).error
    assert not (store.run_dir(run_id) / "report.json").exists()


def test_comparison_rejects_runs_from_different_videos(tmp_path, make_video, monkeypatch, capsys):
    monkeypatch.setattr(Pipeline, "from_config", lambda config: FixedPipeline())
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store)
    ids = []
    for mode, video in (("A", make_video("a.avi", frames=2)), ("B", make_video("b.avi", frames=3))):
        run_id = manager.submit(
            RunConfig(mode=ModeConfig(name=mode), profile=ProfileConfig(name="minimal"), source=video)
        )
        manager.execute(run_id)
        ids.append(run_id)
    monkeypatch.setattr(cli, "RUNS_DIR", store.root)
    with pytest.raises(SystemExit):
        cli.main(["compare", *ids, "--output", str(tmp_path / "comparison.json")])
    assert "video" in capsys.readouterr().err.lower()


def test_mismatched_annotation_pair_fails_before_processing(tmp_path, make_video, monkeypatch):
    video = make_video(frames=1)
    coco, mot = tmp_path / "gt.json", tmp_path / "gt.txt"
    coco.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "width": 64, "height": 48}],
                "categories": [{"id": 1, "name": "person"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 10, 10]}],
            }
        )
    )
    mot.write_text("1,1,2,1,10,10,1,1,1\n")
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store)
    config = RunConfig(
        mode=ModeConfig(name="A"),
        profile=ProfileConfig(name="minimal"),
        source=video,
        annotations=coco,
        tracking_annotations=mot,
    )
    run_id = manager.submit(config)
    manager.execute(run_id)
    assert store.get(run_id).status == RunStatus.FAILED
    assert "do not match" in store.get(run_id).error


def test_annotation_width_mismatch_is_rejected(tmp_path, make_video, monkeypatch):
    video = make_video(frames=1)
    coco, mot = tmp_path / "gt.json", tmp_path / "gt.txt"
    coco.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "width": 65, "height": 48}],
                "categories": [{"id": 1, "name": "person"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 30], "area": 600, "iscrowd": 0}
                ],
            }
        )
    )
    mot.write_text("1,1,10,10,20,30,1,1,1\n")
    monkeypatch.setattr(Pipeline, "from_config", lambda config: FixedPipeline())
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store)
    config = RunConfig(
        mode=ModeConfig(name="A"),
        profile=ProfileConfig(name="minimal"),
        source=video,
        annotations=coco,
        tracking_annotations=mot,
    )
    rejected = manager.submit(config)
    manager.execute(rejected)
    assert store.get(rejected).status == RunStatus.FAILED
    assert "dimensions" in store.get(rejected).error


def test_precision_counts_more_than_100_predictions(tmp_path, make_video, monkeypatch):
    video = make_video(frames=1)
    coco, mot = tmp_path / "gt.json", tmp_path / "gt.txt"
    coco.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "width": 64, "height": 48}],
                "categories": [{"id": 1, "name": "person"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 30], "area": 600, "iscrowd": 0}
                ],
            }
        )
    )
    mot.write_text("1,1,10,10,20,30,1,1,1\n")
    monkeypatch.setattr(Pipeline, "from_config", lambda config: CrowdedPipeline())
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store)
    run_id = manager.submit(
        RunConfig(
            mode=ModeConfig(name="A"),
            profile=ProfileConfig(name="minimal"),
            source=video,
            annotations=coco,
            tracking_annotations=mot,
        )
    )
    manager.execute(run_id)
    assert store.get(run_id).status == RunStatus.COMPLETED, store.get(run_id).error
    assert store.load_metrics(run_id)["precision"] == pytest.approx(1 / 101)


def test_no_predictions_produce_zero_quality_metrics(tmp_path, make_video, monkeypatch):
    video = make_video(frames=1)
    coco, mot = tmp_path / "gt.json", tmp_path / "gt.txt"
    coco.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "width": 64, "height": 48}],
                "categories": [{"id": 1, "name": "person"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 30], "area": 600, "iscrowd": 0}
                ],
            }
        )
    )
    mot.write_text("1,1,10,10,20,30,1,1,1\n")
    monkeypatch.setattr(Pipeline, "from_config", lambda config: EmptyPipeline())
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store)
    run_id = manager.submit(
        RunConfig(
            mode=ModeConfig(name="A"),
            profile=ProfileConfig(name="minimal"),
            source=video,
            annotations=coco,
            tracking_annotations=mot,
        )
    )
    manager.execute(run_id)
    assert store.get(run_id).status == RunStatus.COMPLETED, store.get(run_id).error
    metrics = store.load_metrics(run_id)
    for name in ("map_50_95", "precision", "recall", "hota", "idf1"):
        assert metrics[name] == 0


def test_benchmark_resumes_completed_scenario_and_compares(tmp_path, make_video, monkeypatch):
    video = make_video(frames=1)
    coco, mot = tmp_path / "gt.json", tmp_path / "gt.txt"
    coco.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "width": 64, "height": 48}],
                "categories": [{"id": 1, "name": "person"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 30], "area": 600, "iscrowd": 0}
                ],
            }
        )
    )
    mot.write_text("1,1,10,10,20,30,1,1,1\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "test_cases": [
                    {"file": video.name, "gt_coco": coco.name, "gt_mot": mot.name, "distortion": "none", "level": 0}
                ]
            }
        )
    )
    monkeypatch.setattr(Pipeline, "from_config", lambda config: FixedPipeline())
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)
    cli.main(["benchmark", "--modes", "A"])
    default_progress = tmp_path / "runs" / "benchmark.json"
    assert json.loads(default_progress.read_text())[0]["status"] == "completed"
    progress, comparison = tmp_path / "progress.json", tmp_path / "comparison.json"
    command = ["benchmark", "--manifest", str(manifest), "--scenario", video.name, "--output", str(progress)]
    cli.main(command)
    first = json.loads(progress.read_text())
    cli.main(command)
    assert json.loads(progress.read_text()) == first
    cli.main(["compare", "--benchmark", str(progress), "--manifest", str(manifest), "--output", str(comparison)])
    assert len(json.loads(comparison.read_text())["scenarios"]) == 1
    unpaired_video = make_video("other.avi", frames=2)
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store, recover_interrupted=False)
    unpaired = manager.submit(
        RunConfig(mode=ModeConfig(name="A"), profile=ProfileConfig(name="minimal"), source=unpaired_video)
    )
    manager.execute(unpaired)
    cli.main(
        ["compare", unpaired, "--benchmark", str(progress), "--manifest", str(manifest), "--output", str(comparison)]
    )
    report = json.loads(comparison.read_text())
    assert len(report["scenarios"]) == 1
    assert report["excluded"][0]["run_id"] == unpaired


def test_benchmark_reports_missing_default_manifest(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path / "runs")
    with pytest.raises(SystemExit) as error:
        cli.main(["benchmark"])
    assert error.value.code == 1
    assert "manifest.json" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()
