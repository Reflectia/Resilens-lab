import time

import pytest
import torch

from resilens.core.contracts import StageTiming
from resilens.metrics.telemetry import ResourceSampler, Telemetry
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


def busy(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def test_sampler_measures_ram_and_cpu_of_a_busy_run() -> None:
    sampler = ResourceSampler(cpu_interval_s=0.1)
    sampler.start()
    busy(0.6)
    resources = sampler.stop()

    assert resources["peak_ram_mb"] > 0
    assert resources["peak_cpu_percent"] >= 50  # one busy thread is ~100 %
    assert 30 <= resources["mean_cpu_percent"] <= resources["peak_cpu_percent"] + 1
    assert "peak_gpu_memory_mb" not in resources


def test_sampler_cpu_of_an_idle_run_is_low() -> None:
    sampler = ResourceSampler(cpu_interval_s=0.1)
    sampler.start()
    time.sleep(0.5)

    assert sampler.stop()["mean_cpu_percent"] < 50


def test_sampler_reports_peak_gpu_memory_on_gpu_profile(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 512 * 2**20)
    sampler = ResourceSampler(gpu=True)
    sampler.start()

    assert sampler.stop()["peak_gpu_memory_mb"] == 512
