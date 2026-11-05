import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from resilens.analysis.analyzer import DegradationAnalyzer
from resilens.config.loader import load_mode, load_profile
from resilens.config.schema import ModeConfig, ProfileConfig, RunConfig
from resilens.core.contracts import Detection, FrameContext
from resilens.core.pipeline import NO_PIXELS, Pipeline
from resilens.core.registry import StageRegistry, registry
from resilens.core.stage import Stage
from resilens.detection.yolox import YOLOXNanoDetector
from resilens.detection.yolox_model import yolox_nano
from resilens.enhancement.bypass import BypassEnhancer
from resilens.enhancement.tanet import TANetEnhancer
from resilens.enhancement.tanet_model import TANet
from resilens.io.replay import VideoReplay
from resilens.io.validation import VideoValidator
from resilens.metrics.telemetry import Telemetry
from resilens.policy.engine import PolicyEngine
from resilens.policy.log import export_policy_log, policy_summary
from resilens.testing.mocks import MockDetector, MockTracker, mock_frames
from resilens.tracking.flow_tracker import FlowTracker


def run_config(mode: ModeConfig, profile: str = "minimal") -> RunConfig:
    return RunConfig(mode=mode, profile=load_profile(profile), source=Path("clip.mp4"))


class Recorder(Stage):
    """Logs setup / process / teardown calls; can fail on a given frame."""

    def __init__(self, name: str, log: list[str], fail_on: int | None = None) -> None:
        self.name, self.log, self.fail_on = name, log, fail_on

    def setup(self) -> None:
        self.log.append(f"setup {self.name}")

    def process(self, ctx: FrameContext) -> FrameContext:
        if ctx.index == self.fail_on:
            raise RuntimeError("stage failed")
        self.log.append(f"{self.name} {ctx.index}")
        return ctx

    def teardown(self) -> None:
        self.log.append(f"teardown {self.name}")


def test_registry_creates_stage_with_config_params() -> None:
    stages = StageRegistry()
    stages.register("recorder")(Recorder)

    stage = stages.create("recorder", name="r", log=[])

    assert isinstance(stage, Recorder) and stage.name == "r"
    assert stages.names() == ["recorder"]


def test_registry_rejects_unknown_and_duplicate_names() -> None:
    stages = StageRegistry()
    stages.register("recorder")(Recorder)
    stages.register("recorder")(Recorder)  # importing a module twice is harmless

    with pytest.raises(ValueError, match="Unknown stage 'missing'. Available: recorder"):
        stages.create("missing")
    with pytest.raises(ValueError, match="already registered"):
        stages.register("recorder")(MockDetector)


def test_stages_used_in_mode_configs_are_registered() -> None:
    assert {"bypass", "tanet", "yolox_nano", "flow"} <= set(registry.names())


def test_mode_a_builds_enhancer_detector_tracker_without_loading_weights() -> None:
    pipeline = Pipeline.from_config(run_config(load_mode("A")))

    enhancer, detector, tracker = pipeline.stages
    assert isinstance(enhancer, BypassEnhancer)
    assert isinstance(detector, YOLOXNanoDetector) and detector.model is None
    assert isinstance(tracker, FlowTracker) and tracker.targets is None
    assert detector.class_ids == [0]
    assert detector.device == "cpu"
    assert pipeline.telemetry.warmup_frames == 10


def test_profile_device_is_passed_unless_mode_sets_it() -> None:
    mode = ModeConfig(name="X", detector_params={"weights_path": "w.pth"}, tracker="flow")
    gpu = ProfileConfig(name="gpu", device="cuda")

    assert Pipeline.from_config(RunConfig(mode=mode, profile=gpu, source=Path("v.mp4"))).stages[1].device == "cuda"
    mode.detector_params["device"] = "cpu"
    assert Pipeline.from_config(RunConfig(mode=mode, profile=gpu, source=Path("v.mp4"))).stages[1].device == "cpu"


def test_analyzer_runs_first_when_mode_uses_it() -> None:
    mode = ModeConfig(name="X", use_analyzer=True, detector_params={"weights_path": "w.pth"}, tracker="flow")

    assert isinstance(Pipeline.from_config(run_config(mode)).stages[0], DegradationAnalyzer)


def test_mode_d_puts_policy_between_analyzer_and_enhancer() -> None:
    pipeline = Pipeline.from_config(run_config(load_mode("D")))

    analyzer, policy, enhancer, detector, tracker = pipeline.stages
    assert isinstance(analyzer, DegradationAnalyzer)
    assert isinstance(policy, PolicyEngine) and policy.profile.name == "minimal"
    assert isinstance(enhancer, TANetEnhancer) and enhancer.model is None
    assert policy.budget_source == pipeline.telemetry.budget_state
    assert pipeline.telemetry.latency_metric == policy.rules.config.budget.latency_metric


def test_policy_needs_the_analyzer() -> None:
    mode = ModeConfig(name="X", use_policy=True, detector_params={"weights_path": "w.pth"}, tracker="flow")

    with pytest.raises(ValueError, match="needs the analyzer"):
        Pipeline.from_config(run_config(mode))


def save_random_weights(tmp_path: Path) -> tuple[Path, Path]:
    torch.manual_seed(0)
    yolox, tanet = tmp_path / "yolox_nano.pth", tmp_path / "tanet.pth"
    torch.save({"model": yolox_nano().state_dict()}, yolox)
    torch.save(TANet().state_dict(), tanet)
    return yolox, tanet


def test_mode_d_on_video_logs_a_decision_per_frame(tmp_path: Path, make_video) -> None:
    """Analyzer -> policy -> TANet -> YOLOX-Nano -> flow tracker, random weights, on a real video file."""
    yolox, tanet = save_random_weights(tmp_path)
    mode = load_mode("D")
    mode.enhancer_params["weights_path"] = tanet
    mode.detector_params["weights_path"] = yolox
    video = VideoValidator().validate(make_video("clip.mp4", frames=4, width=128, height=96))

    frames = Pipeline.from_config(run_config(mode)).run(VideoReplay(video))
    export_policy_log(frames, "minimal", tmp_path / "policy_log.csv")

    assert [t.stage for t in frames[0].timings] == ["replay", "analyzer", "policy", "tanet", "yolox_nano", "flow"]
    assert all(f.decision is not None and f.degradation is not None for f in frames)
    with (tmp_path / "policy_log.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["frame"] for row in rows] == ["1", "2", "3", "4"]
    assert {row["profile"] for row in rows} == {"minimal"}
    assert rows[0]["action"] == frames[0].decision.action
    assert rows[0]["ram_mb"] != ""
    summary = policy_summary(frames)
    action_shares = [summary[f"policy_{action}_share"] for action in ("enhance", "skip", "bypass", "downscale")]
    assert sum(action_shares) == pytest.approx(1.0)


def test_frames_over_the_latency_budget_are_downgraded(tmp_path: Path, make_video) -> None:
    yolox, _ = save_random_weights(tmp_path)
    mode = ModeConfig(name="X", use_analyzer=True, use_policy=True, detector_params={"weights_path": yolox})
    mode.tracker = "flow"
    profile = load_profile("minimal").model_copy(update={"frame_budget_ms": 0.001, "downscale_resolution": (64, 48)})
    config = RunConfig(mode=mode, profile=profile, source=Path("clip.mp4"), warmup_frames=0)
    video = VideoValidator().validate(make_video("clip.mp4", frames=3, width=128, height=96))

    frames = Pipeline.from_config(config).run(VideoReplay(video))

    assert not frames[0].decision.downgraded  # no latency measured before the first frame
    assert all(f.decision.downgraded for f in frames[1:])
    assert all("ms > budget" in f.decision.reason for f in frames[1:])


def test_unknown_stage_in_config() -> None:
    with pytest.raises(ValueError, match="Unknown stage 'bytetrack'"):
        Pipeline.from_config(run_config(ModeConfig(name="X", detector_params={"weights_path": "w.pth"})))


def test_stages_run_in_order_with_timings() -> None:
    pipeline = Pipeline([MockDetector(), MockTracker()], Telemetry())

    ctx = pipeline.process_frame(next(mock_frames(count=1)))

    assert [t.stage for t in ctx.timings] == ["mock_detector", "mock_tracker"]
    assert all(t.duration_ms >= 0 for t in ctx.timings)
    assert ctx.tracks[0].bbox == ctx.detections[0].bbox
    assert pipeline.telemetry.latencies_ms == [sum(t.duration_ms for t in ctx.timings)]


def test_run_sets_up_once_keeps_results_and_drops_pixels() -> None:
    log: list[str] = []
    pipeline = Pipeline([Recorder("a", log), MockDetector()], Telemetry())

    frames = pipeline.run(mock_frames(count=2))

    assert log == ["setup a", "a 0", "a 1", "teardown a"]
    assert [f.index for f in frames] == [0, 1]
    assert frames[0].detections == [Detection(bbox=(10.0, 10.0, 30.0, 40.0), score=0.9, class_id=0)]
    assert frames[0].image is NO_PIXELS and frames[0].enhanced is None


def test_failing_stage_still_tears_down_and_keeps_its_timing() -> None:
    log: list[str] = []
    pipeline = Pipeline([Recorder("a", log, fail_on=1)], Telemetry())
    frames = list(mock_frames(count=2))

    with pytest.raises(RuntimeError, match="stage failed"):
        pipeline.run(frames)

    assert log[-1] == "teardown a"
    assert [t.stage for t in frames[1].timings] == ["a"]


def test_mode_a_on_video_with_real_stages(tmp_path: Path, make_video) -> None:
    """Bypass -> YOLOX-Nano (random weights) -> flow tracker on a real video file."""
    weights = tmp_path / "yolox_nano.pth"
    torch.manual_seed(0)
    torch.save({"model": yolox_nano().state_dict()}, weights)
    mode = load_mode("A")
    # random weights: low scores, and "person" is rarely the top class
    mode.detector_params |= {"weights_path": weights, "conf_threshold": 0.0, "class_ids": None}
    video = VideoValidator().validate(make_video("clip.mp4", frames=4, width=128, height=96))

    frames = Pipeline.from_config(run_config(mode)).run(VideoReplay(video))

    assert len(frames) == 4
    assert [t.stage for t in frames[0].timings] == ["replay", "bypass", "yolox_nano", "flow"]
    assert frames[0].detections
    first_ids = {t.track_id for t in frames[0].tracks}
    assert first_ids == set(range(1, len(frames[0].detections) + 1))
    assert {t.track_id for t in frames[-1].tracks} <= first_ids


def test_no_pixels_is_an_empty_image() -> None:
    assert NO_PIXELS.size == 0 and NO_PIXELS.dtype == np.uint8
