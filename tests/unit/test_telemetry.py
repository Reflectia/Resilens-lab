import pytest

from resilens.core.contracts import StageTiming
from resilens.metrics.telemetry import Telemetry
from resilens.testing.mocks import mock_frames


def record(telemetry: Telemetry, *latencies_ms: float) -> None:
    for ctx, latency_ms in zip(mock_frames(len(latencies_ms)), latencies_ms, strict=True):
        ctx.timings = [StageTiming("replay", 1.0), StageTiming("detector", latency_ms - 1.0)]
        telemetry.record_frame(ctx)


def test_frame_latency_is_the_sum_of_its_stages() -> None:
    telemetry = Telemetry(warmup_frames=0)

    record(telemetry, 30.0, 50.0)

    assert telemetry.latencies_ms == [30.0, 50.0]


def test_no_latency_before_warmup_ends() -> None:
    telemetry = Telemetry(warmup_frames=2)

    record(telemetry, 500.0, 400.0)

    assert telemetry.recent_latency_ms() is None
    record(telemetry, 20.0)
    assert telemetry.recent_latency_ms() == 20.0


def test_moving_mean_over_the_window() -> None:
    telemetry = Telemetry(warmup_frames=0, latency_window=3)

    record(telemetry, 100.0, 10.0, 20.0, 30.0)

    assert telemetry.recent_latency_ms() == pytest.approx(20.0)


def test_moving_p95() -> None:
    telemetry = Telemetry(warmup_frames=0, latency_window=20, latency_metric="p95")

    record(telemetry, *[float(ms) for ms in range(1, 21)])

    assert telemetry.recent_latency_ms() == pytest.approx(19.05)  # numpy linear interpolation


def test_budget_state_measures_this_process() -> None:
    telemetry = Telemetry(warmup_frames=0)
    record(telemetry, 40.0)

    budget = telemetry.budget_state()

    assert budget.latency_ms == 40.0
    assert budget.ram_mb > 0
    assert budget.cpu_percent >= 0
    assert budget.gpu_memory_mb is None  # not a GPU profile
