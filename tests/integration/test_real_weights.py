"""Opt-in end-to-end checks with trained weights and a real control video."""

import csv
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch

from resilens.analysis.analyzer import DegradationAnalyzer
from resilens.config.loader import build_run_config
from resilens.config.schema import RunConfig
from resilens.core.contracts import Action, DegradationEstimate, DegradationType
from resilens.enhancement.tanet import TANetEnhancer
from resilens.io.validation import VideoValidator
from resilens.policy.rules import RuleTable
from resilens.runs.manager import RunManager
from resilens.runs.models import RunRecord, RunStatus
from resilens.runs.store import RunStore

pytestmark = [pytest.mark.integration, pytest.mark.real_weights]


@pytest.fixture
def video(request: pytest.FixtureRequest) -> Path:
    value = request.config.getoption("--test-video")
    if not value or not Path(value).is_file():
        pytest.skip(f"Control video unavailable: {value or 'pass --test-video'}")
    return Path(value).resolve()


def configure(request: pytest.FixtureRequest, mode: str, profile: str, video: Path) -> RunConfig:
    config = build_run_config(mode, profile, video)
    weights = Path(request.config.getoption("--weights-dir")).resolve()
    config.mode.detector_params["weights_path"] = str(weights / "yolox_nano_coco.pth")
    if mode == "D":
        config.mode.enhancer_params["weights_path"] = str(weights / "tanet.pth")
    return config


def require_assets(config: RunConfig) -> None:
    paths = [config.mode.detector_params["weights_path"]]
    if config.mode.name == "D":
        paths.append(config.mode.enhancer_params["weights_path"])
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        pytest.skip(f"Missing trained weights: {', '.join(missing)}")
    if config.profile.device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable in this PyTorch environment")


def execute(config: RunConfig, tmp_path: Path) -> tuple[RunStore, str, RunRecord]:
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store, recover_interrupted=False)
    run_id = manager.submit(config)
    manager.execute(run_id)
    return store, run_id, store.get(run_id)


def assert_completed(store: RunStore, run_id: str, record: RunRecord, config: RunConfig) -> dict[str, float]:
    assert record.status == RunStatus.COMPLETED, record.error
    directory = store.run_dir(run_id)
    metrics = store.load_metrics(run_id)
    report = store.load_report(run_id)
    frame_count = VideoValidator().validate(config.source).frame_count
    assert metrics["processed_frames"] == frame_count
    assert metrics["frames_dropped"] == 0
    assert metrics["fps"] > 0 and metrics["peak_ram_mb"] > 0
    assert report["run_id"] == run_id and report["mode"] == config.mode.name
    assert report["profile"] == config.profile.name and report["metrics"] == metrics
    assert store.load_config(run_id).mode.name == config.mode.name
    for name in ("config.json", "detections.txt", "tracks.txt", "report.json"):
        assert (directory / name).is_file()
    return metrics


def policy_rows(store: RunStore, run_id: str) -> list[dict[str, str]]:
    with (store.run_dir(run_id) / "policy_log.csv").open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


@pytest.mark.parametrize("profile", ["minimal", "standard", "performance"])
@pytest.mark.parametrize("case", ["A", "bypass", "enhance"])
def test_real_pipeline_profiles(
    request: pytest.FixtureRequest,
    video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    case: str,
) -> None:
    config = configure(request, "A" if case == "A" else "D", profile, video)
    require_assets(config)
    calls: list[bool] = []
    original_enhance = TANetEnhancer.enhance

    def observe_enhance(self: TANetEnhancer, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        calls.append(True)
        return original_enhance(self, image)

    monkeypatch.setattr(TANetEnhancer, "enhance", observe_enhance)
    if case != "A":
        kind = DegradationType.NONE if case == "bypass" else DegradationType.FOG
        level = 0 if case == "bypass" else 2
        monkeypatch.setattr(DegradationAnalyzer, "estimate", lambda self, ctx: DegradationEstimate(kind, level, 1.0))
        if case == "enhance":
            rules = RuleTable.load()
            for rule in rules.config.rules:
                if DegradationType.FOG in rule.types and rule.level == 2:
                    rule.actions[profile] = Action.ENHANCE
            monkeypatch.setattr("resilens.core.pipeline.RuleTable.load", lambda: RuleTable(rules.config))
    store, run_id, record = execute(config, tmp_path)
    metrics = assert_completed(store, run_id, record, config)
    if case == "A":
        assert not calls
        return
    rows = policy_rows(store, run_id)
    assert len(rows) == metrics["processed_frames"]
    assert all(row["base_action"] == case for row in rows)
    assert all(row["action"] == case for row in rows if row["downgraded"] == "0")
    enhanced = sum(row["enhanced"] == "1" for row in rows)
    assert len(calls) == enhanced
    assert enhanced > 0 if case == "enhance" else enhanced == 0


def test_real_analyzer_policy(request: pytest.FixtureRequest, video: Path, tmp_path: Path) -> None:
    config = configure(request, "D", "standard", video)
    require_assets(config)
    store, run_id, record = execute(config, tmp_path)
    metrics = assert_completed(store, run_id, record, config)
    rows = policy_rows(store, run_id)
    rules = RuleTable.load()
    assert len(rows) == metrics["processed_frames"]
    for row in rows:
        expected = rules.lookup(DegradationType(row["degradation"]), int(row["level"]), "standard")
        assert row["base_action"] == expected
        if row["downgraded"] == "0":
            assert row["action"] == expected


@pytest.mark.parametrize("mode,missing", [("A", "detector"), ("D", "detector"), ("D", "enhancer")])
def test_missing_weight_fails(
    request: pytest.FixtureRequest, video: Path, tmp_path: Path, mode: str, missing: str
) -> None:
    config = configure(request, mode, "standard", video)
    require_assets(config)
    params = config.mode.detector_params if missing == "detector" else config.mode.enhancer_params
    params["weights_path"] = str(tmp_path / "missing-weights.pth")
    _, _, record = execute(config, tmp_path)
    assert record.status == RunStatus.FAILED
    assert record.error and "missing-weights.pth" in record.error


def test_invalid_video_fails(request: pytest.FixtureRequest, tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.mp4"
    invalid.write_bytes(b"not a video")
    config = configure(request, "A", "standard", invalid)
    _, _, record = execute(config, tmp_path)
    assert record.status == RunStatus.FAILED
    assert record.error and "video" in record.error.lower()
