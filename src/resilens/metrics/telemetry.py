import time
from collections import deque
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from threading import Event, Thread
from typing import Literal

import numpy as np
import psutil
import torch

from resilens.core.contracts import BudgetState, FrameContext, StageTiming
from resilens.metrics.latency import LatencySummary, frame_latency_ms, summarize_latencies


@contextmanager
def _stage_timer(ctx: FrameContext, stage: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        ctx.timings.append(StageTiming(stage, (time.perf_counter() - started) * 1000))


class ResourceSampler:
    """Observe peak process RAM while a run is executing."""

    def __init__(self, interval_s: float = 0.01) -> None:
        self.interval_s = interval_s
        self._process = psutil.Process()
        self._stopping = Event()
        self._thread: Thread | None = None
        self._peak_ram_mb = 0.0

    def start(self) -> None:
        self._peak_ram_mb = 0.0
        self._stopping.clear()
        self._sample()
        self._thread = Thread(target=self._poll, name="resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        """Return peak values collected since start()."""
        self._stopping.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._sample()
        return {"peak_ram_mb": self._peak_ram_mb}

    def _sample(self) -> None:
        self._peak_ram_mb = max(self._peak_ram_mb, self._process.memory_info().rss / 2**20)

    def _poll(self) -> None:
        while not self._stopping.wait(self.interval_s):
            self._sample()


class Telemetry:
    """Per-frame latency of a run and the live budget state the Policy Engine decides on.

    The first `warmup_frames` frames (model warm-up) do not count towards the moving latency.
    """

    def __init__(
        self,
        warmup_frames: int = 10,
        latency_window: int = 30,
        latency_metric: Literal["mean", "p95"] = "mean",
        gpu: bool = False,
    ) -> None:
        self.warmup_frames = warmup_frames
        self.latency_metric = latency_metric
        self.gpu = gpu
        self.resources = ResourceSampler()
        self.latencies_ms: list[float] = []
        self._recent_ms: deque[float] = deque(maxlen=latency_window)
        self._process = psutil.Process()
        self._process.cpu_percent(None)  # the first call only starts the measurement

    def stage_timer(self, ctx: FrameContext, stage: str) -> AbstractContextManager[None]:
        """Append the time spent inside the `with` block to ctx.timings, also when the stage raises."""
        return _stage_timer(ctx, stage)

    def record_frame(self, ctx: FrameContext) -> None:
        """Store end-to-end latency (sum of ctx.timings, frame reading included) of a finished frame."""
        latency_ms = frame_latency_ms(ctx)
        self.latencies_ms.append(latency_ms)
        if len(self.latencies_ms) > self.warmup_frames:
            self._recent_ms.append(latency_ms)

    def recent_latency_ms(self) -> float | None:
        """Mean or p95 latency of the last frames after warm-up; None before the first one."""
        if not self._recent_ms:
            return None
        recent = np.array(self._recent_ms)
        return float(recent.mean() if self.latency_metric == "mean" else np.percentile(recent, 95))

    def budget_state(self) -> BudgetState:
        """Moving latency and current RAM, CPU and (on a GPU profile) GPU memory of this process."""
        gpu_memory_mb = torch.cuda.memory_allocated() / 2**20 if self.gpu and torch.cuda.is_available() else None
        return BudgetState(
            latency_ms=self.recent_latency_ms(),
            ram_mb=self._process.memory_info().rss / 2**20,
            cpu_percent=self._process.cpu_percent(None),
            gpu_memory_mb=gpu_memory_mb,
        )

    def summary(self, frame_budget_ms: float | None = None) -> LatencySummary:
        """FPS, p50 / p95 and the share of frames within the budget, after warm-up."""
        return summarize_latencies(self.latencies_ms, frame_budget_ms, self.warmup_frames)
