import pytest

from resilens.core.contracts import StageTiming
from resilens.metrics.latency import latency_metrics, summarize_latencies
from resilens.metrics.telemetry import Telemetry
from resilens.testing.mocks import mock_frames


def frames_with(*stage_ms: dict[str, float]):
    frames = list(mock_frames(len(stage_ms)))
    for ctx, stages in zip(frames, stage_ms, strict=True):
        ctx.timings = [StageTiming(stage, ms) for stage, ms in stages.items()]
    return frames


def test_percentiles_fps_and_budget_share() -> None:
    summary = summarize_latencies([float(ms) for ms in range(1, 21)], frame_budget_ms=10.0)

    assert summary.p50_ms == pytest.approx(10.5)
    assert summary.p95_ms == pytest.approx(19.05)  # numpy linear interpolation
    assert summary.fps == pytest.approx(1000 / 10.5)
    assert summary.in_budget_ratio == pytest.approx(0.5)


def test_no_budget_share_without_a_budget() -> None:
    assert summarize_latencies([10.0], frame_budget_ms=None).in_budget_ratio is None


def test_warmup_frames_are_left_out() -> None:
    summary = summarize_latencies([900.0, 800.0, 10.0, 20.0], frame_budget_ms=None, warmup_frames=2)

    assert summary.p50_ms == pytest.approx(15.0)


def test_run_shorter_than_warmup_uses_all_frames() -> None:
    assert summarize_latencies([10.0, 30.0], frame_budget_ms=None, warmup_frames=10).p50_ms == pytest.approx(20.0)


def test_no_latencies_is_an_error() -> None:
    with pytest.raises(ValueError, match="No frame latencies"):
        summarize_latencies([], frame_budget_ms=None)


def test_run_metrics_cover_end_to_end_and_each_stage() -> None:
    frames = frames_with(
        {"replay": 500.0, "yolox_nano": 500.0},  # warm-up
        {"replay": 2.0, "yolox_nano": 30.0},
        {"replay": 4.0, "yolox_nano": 50.0},
    )

    metrics = latency_metrics(frames, frame_budget_ms=40.0, warmup_frames=1)

    assert metrics["latency_p50_ms"] == pytest.approx(43.0)
    assert metrics["latency_in_budget_ratio"] == pytest.approx(0.5)
    assert metrics["stage_replay_p50_ms"] == pytest.approx(3.0)
    assert metrics["stage_yolox_nano_p95_ms"] == pytest.approx(49.0)
    assert "latency_in_budget_ratio" not in latency_metrics(frames, frame_budget_ms=None, warmup_frames=1)


def test_telemetry_summary_uses_recorded_frames_after_warmup() -> None:
    telemetry = Telemetry(warmup_frames=1)
    for ctx in frames_with({"replay": 500.0}, {"replay": 20.0}, {"replay": 40.0}):
        telemetry.record_frame(ctx)

    assert telemetry.summary(frame_budget_ms=30.0).in_budget_ratio == pytest.approx(0.5)
